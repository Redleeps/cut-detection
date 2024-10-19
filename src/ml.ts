import {
  IMAGE_BYTE_SIZE,
  IMAGE_SIZE,
  PLAN_INDEX_BYTE_SIZE,
} from "./constantes";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { askBoolean, chunkArray, joinBuffers, mean, random } from "./utils";
import formatDuration from "format-duration";
import { openDatasetSync, rawBufferToPng } from "./dataset";
import { readSync } from "fs";
import path from "path";
import { cwd } from "process";
import { mkdir, writeFile } from "fs/promises";
import sharp from "sharp";
import { SpatialPyramidPooling } from "./sppf";
import { C3 } from "./c3";

export async function loadModel(name: string, compileModel = false) {
  const model = await tf.loadLayersModel(
    "file://" + path.join(cwd(), "models", name, "model.json"),
  );
  console.log(`Model "${name}" loaded with ${model.countParams()} parameters`);
  return model;
}

export function createSequentialModel(name: string) {
  const model = tf.sequential({
    name,
    layers: [
      tf.layers.rescaling({
        scale: 1 / 255,
        inputShape: [IMAGE_SIZE.height * 5, IMAGE_SIZE.width, 1],
      }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({
        filters: 8,
        kernelSize: 9,
        activation: "relu",
      }),
      tf.layers.conv2d({
        filters: 16,
        kernelSize: 9,
        activation: "relu",
      }),
      tf.layers.batchNormalization(),
      tf.layers.maxPooling2d({ poolSize: 4 }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: "relu" }),
      tf.layers.dense({ units: 1, activation: "sigmoid" }),
    ],
  });
  compile(model);
  console.log(`Model "${name}" created with ${model.countParams()} parameters`);
  return model;
}
function convBlock(filters: number, kernelSize: number, strides: number) {
  return tf.layers.conv2d({
    filters,
    kernelSize,
    strides,
    activation: "relu",
    padding: "same",
  });
}

export function createModel(name: string) {
  const inputs = tf.layers.input({
    shape: [IMAGE_SIZE.height * 5, IMAGE_SIZE.width, 1],
  });
  const rescaling = tf.layers
    .rescaling({
      scale: 1 / 255,
    })
    .apply(inputs) as tf.SymbolicTensor;
  // Stage 1
  let x = convBlock(64, 6, 2).apply(rescaling) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;

  // Stage 2
  x = convBlock(128, 3, 2).apply(x) as tf.SymbolicTensor;
  x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
  x = new C3(128).apply(x) as tf.SymbolicTensor;

  x = new SpatialPyramidPooling({
    bins: [4],
  }).apply(x) as tf.SymbolicTensor;

  const flatten = tf.layers.flatten().apply(x) as tf.SymbolicTensor;

  const dense = tf.layers
    .dense({ units: 128, activation: "relu" })
    .apply(flatten) as tf.SymbolicTensor;

  const outputs = tf.layers
    .dense({ units: 1, activation: "sigmoid" })
    .apply(dense) as tf.SymbolicTensor;

  const model = tf.model({
    name,
    inputs,
    outputs,
  });

  compile(model);

  console.log(`Model "${name}" created with ${model.countParams()} parameters`);
  return model;
}
export function createMultiInputModel(name: string) {
  const head = tf.sequential({
    layers: [
      tf.layers.rescaling({
        scale: 1 / 255,
        inputShape: [IMAGE_SIZE.height, IMAGE_SIZE.width, 1],
      }),
      convBlock(64, 6, 2),
      tf.layers.batchNormalization(),
      convBlock(128, 3, 2),
      tf.layers.batchNormalization(),
      new C3(128),
      new SpatialPyramidPooling({
        bins: [4],
      }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: "relu" }),
    ],
  });
  compile(head);
  const body = tf.sequential({
    name,
    layers: [
      tf.layers.timeDistributed({
        layer: head,
        inputShape: [5, IMAGE_SIZE.height, IMAGE_SIZE.width, 1],
      }),
      tf.layers.gru({ units: 64 }),
      tf.layers.dense({ units: 64, activation: "relu" }),
      tf.layers.dense({ units: 1, activation: "sigmoid" }),
    ],
  });

  compile(body);

  console.log(`Model "${name}" created with ${body.countParams()} parameters`);
  return body;
}

export async function compile(
  model: tf.LayersModel | tf.Sequential,
  learningRate = 0.001,
) {
  model.compile({
    loss: "binaryCrossentropy",
    optimizer: new tf.AdamOptimizer(learningRate, 0.9, 0.999, 1e-7),
    metrics: ["accuracy"],
  });
}

export interface ITrainingModelParams<T> {
  model: tf.LayersModel;
  epochs: number;
  dataset?: string;
  fitThreshlod?: number;
  multiInput?: boolean;
  iterator: (
    params: IGenerateDatasetIteratorParams,
  ) => Parameters<typeof tf.data.generator>[0];
}
export async function trainModel<T>({
  model,
  epochs,
  dataset: datasetName,
  fitThreshlod = 0.999,
  iterator,
}: ITrainingModelParams<T>) {
  console.log(
    `Training ${model.name} on ${datasetName || "dataset"} with ${epochs} epochs and fitness threshold of ${fitThreshlod}`,
  );
  const engine = tf.engine();
  engine.startScope();

  const batchSize = 128;
  console.log("creating tensors");
  using dataset = openDatasetSync(datasetName || "dataset");
  const trainBatchCount = Math.floor((dataset.length * 10) / batchSize);
  const trainingData = tf.data
    .generator(
      //@ts-ignore
      iterator({
        fileHandler: dataset.handler,
        batchSize,
        blockOffset: 0,
        blockCount: dataset.length,
        batchCount: trainBatchCount,
      }),
    )
    .repeat(epochs);
  const validationData = tf.data
    .generator(
      //@ts-ignore
      iterator({
        fileHandler: dataset.handler,
        batchSize,
        blockOffset: 0,
        blockCount: dataset.length,
        batchCount: Math.floor(dataset.length / batchSize / 10),
      }),
    )
    .repeat(epochs);
  if (!model.getUserDefinedMetadata()) {
    console.log("no user defined metadata");
    model.setUserDefinedMetadata({
      acc: 0,
      loss: -1 + 2 ** 32,
    });
  }
  console.log("Training with dataset...\n");
  const stopTraining = () => {
    console.log("Waiting for an eventual batch to complete");
    model.stopTraining = true;
  };
  process.once("SIGINT", stopTraining);
  await model.fitDataset(trainingData, {
    batchesPerEpoch: trainBatchCount,
    validationBatchSize: batchSize,
    validationBatches: Math.floor(dataset.length / batchSize / 10),
    validationData,
    epochs,
    callbacks: {
      async onEpochEnd(_epoch, perfs) {
        if (!perfs) return;
        const bestPerfs = (model.getUserDefinedMetadata() ?? {}) as {
          loss: number;
          acc: number;
        };
        bestPerfs.acc ??= 0;
        bestPerfs.loss ??= Infinity;
        if (
          perfs.val_acc > bestPerfs.acc ||
          (perfs.val_acc === bestPerfs.acc && perfs.val_loss < bestPerfs.loss)
        ) {
          model.setUserDefinedMetadata({
            ...bestPerfs,
            acc: perfs.val_acc,
            loss: perfs.val_loss,
          });
          await saveModel(model);
        }
        if (perfs.acc < fitThreshlod || perfs.val_acc < fitThreshlod) return;
        if (!model.stopTraining) return;
        console.log("Model is trained enough, stopping training.");
        stopTraining();
      },
    },
  });
  process.off("SIGINT", stopTraining);
  engine.endScope();
  console.log("Training done");
  const shouldSaveModel = await askBoolean(
    "Do you want to save the model? (Y/n) (15s to answer): ",
    15_000,
  );
  if (shouldSaveModel) await saveModel(model);
  return model;
}
export async function saveModel(model: tf.LayersModel | tf.Sequential) {
  const savePath = path.join(cwd(), "models", model.name);
  const historySavePath = path.join(savePath, "history", Date.now() + "");
  await mkdir(historySavePath, { recursive: true }).catch(console.error);
  await model.save("file://" + historySavePath);
  await model.save("file://" + savePath);
  console.log("**Model saved**");
}
export async function benchModel(
  model: tf.LayersModel,
  iterator: (typeof ITERATORS)[keyof typeof ITERATORS],
  datasetName?: string,
) {
  console.log(`Benchmarking ${model.name} on ${datasetName || "dataset"}`);
  using dataset = openDatasetSync(datasetName || "dataset");
  const batchSize = 16;
  const batchCount = 10 * Math.ceil(dataset.length / batchSize);
  const batches = iterator({
    fileHandler: dataset.handler,
    batchSize,
    blockOffset: 0,
    blockCount: dataset.length,
    batchCount,
  });
  console.log("Starting benchmark");
  let totalLoss: number[] = [],
    totalAccuracy: number[] = [];
  const start = performance.now();

  await mkdir("test", { recursive: true }).catch(() => null);
  for (const batch of batches()) {
    const result = model.evaluate(batch.xs, batch.ys) as tf.Tensor<tf.Rank>[];
    const [loss, accuracy] = result.map((t) => t.arraySync() as number);

    totalLoss.push(loss);
    totalAccuracy.push(accuracy);

    const meanLoss = mean(totalLoss);
    const meanAccuracy = mean(totalAccuracy);
    process.stdout.write(
      `\rAccuracy=${accuracy.toPrecision(4)}(µ=${meanAccuracy.toPrecision(4)}), Loss=${loss.toPrecision(3).padStart(12, "*")}(µ=${meanLoss.toPrecision(3)}), Batch=${batch.batch + 1}/${batchCount}, Duration=${formatDuration(performance.now() - start, { ms: true })}`,
    );
  }
  console.log();
}

function pickImageSequenceF(
  file: Buffer,
  batch: number,
  count: number,
  offset: number,
  set?: Set<string>,
) {
  const blockLength = IMAGE_BYTE_SIZE + PLAN_INDEX_BYTE_SIZE;
  function fetchAtIndex(index: number) {
    const cursor = (offset + index) * blockLength + 4;
    const outputBuffer = file.subarray(cursor, cursor + PLAN_INDEX_BYTE_SIZE);
    const inputBuffer = file.subarray(
      cursor + PLAN_INDEX_BYTE_SIZE,
      cursor + PLAN_INDEX_BYTE_SIZE + IMAGE_BYTE_SIZE,
    );
    return [inputBuffer, outputBuffer.readUInt32LE(0)] as const;
  }

  const xBuffers: Buffer[] = [];
  let planIndex = -1;
  let isSamePlan = true;

  fistHalf: {
    let randomOffset = Math.floor(random() * count);

    const first = fetchAtIndex(randomOffset);
    xBuffers.push(first[0]);
    planIndex = first[1];

    let second = fetchAtIndex(randomOffset + 1);
    if (second[1] === planIndex) xBuffers.push(second[0]);
    else xBuffers.unshift(fetchAtIndex(randomOffset - 1)[0]);

    let third = fetchAtIndex(randomOffset + 2);
    if (third[1] === planIndex) xBuffers.push(third[0]);
    else xBuffers.unshift(fetchAtIndex(randomOffset - 2)[0]);
    if (!(batch % 3)) {
      let fourth = fetchAtIndex(randomOffset + 3);
      if (fourth[1] === planIndex) xBuffers.push(fourth[0]);
      else xBuffers.unshift(fetchAtIndex(randomOffset - 3)[0]);
      if (!(batch % 4)) {
        let fifth = fetchAtIndex(randomOffset + 4);
        if (fifth[1] === planIndex) xBuffers.push(fifth[0]);
        else xBuffers.unshift(fetchAtIndex(randomOffset - 4)[0]);
        set?.add(`${planIndex}:${planIndex}`);
      }
    }
  }

  secondHalf: {
    if (!(batch % 3) && !(batch % 4)) break secondHalf;
    let randomOffset = Math.floor(random() * count);
    let newPlanIndex = -1;
    let tries = 100;
    do {
      let randomOffset = Math.floor(random() * count);
      [, newPlanIndex] = fetchAtIndex(randomOffset);
      tries--;
    } while (
      tries &&
      (newPlanIndex === planIndex || set?.has(`${planIndex}:${newPlanIndex}`))
    );
    set?.add(`${planIndex}:${newPlanIndex}`);

    const first = fetchAtIndex(randomOffset);
    const push = batch % 5 ? "push" : "unshift";
    if (!(batch % 3)) {
      xBuffers[push](first[0]);
      break secondHalf;
    }

    isSamePlan = batch % 5 ? planIndex === first[1] : false;

    let second = fetchAtIndex(randomOffset + 1);
    if (second[1] === first[1]) {
      xBuffers[push](first[0]);
      xBuffers[push](second[0]);
    } else {
      xBuffers[push](fetchAtIndex(randomOffset - 1)[0]);
      xBuffers[push](first[0]);
    }
  }

  return [xBuffers, isSamePlan, planIndex] as const;
}
function pickImageSequence(
  fileHandler: number,
  batch: number,
  count: number,
  offset: number,
  set?: Set<string>,
) {
  const blockLength = IMAGE_BYTE_SIZE + PLAN_INDEX_BYTE_SIZE;
  function fetchAtIndex(index: number) {
    const inputBuffer = Buffer.alloc(IMAGE_BYTE_SIZE);
    const outputBuffer = Buffer.alloc(PLAN_INDEX_BYTE_SIZE);
    const cursor = (offset + index) * blockLength + 4;
    readSync(fileHandler, outputBuffer, 0, PLAN_INDEX_BYTE_SIZE, cursor);
    readSync(
      fileHandler,
      inputBuffer,
      0,
      IMAGE_BYTE_SIZE,
      cursor + PLAN_INDEX_BYTE_SIZE,
    );
    return [inputBuffer, outputBuffer.readUInt32LE(0)] as const;
  }

  const xBuffers: Buffer[] = [];
  let planIndex = -1;
  let isSamePlan = true;

  fistHalf: {
    let randomOffset = Math.floor(random() * count);

    const first = fetchAtIndex(randomOffset);
    xBuffers.push(first[0]);
    planIndex = first[1];

    let second = fetchAtIndex(randomOffset + 1);
    if (second[1] === planIndex) xBuffers.push(second[0]);
    else xBuffers.unshift(fetchAtIndex(randomOffset - 1)[0]);

    let third = fetchAtIndex(randomOffset + 2);
    if (third[1] === planIndex) xBuffers.push(third[0]);
    else xBuffers.unshift(fetchAtIndex(randomOffset - 2)[0]);
    if (!(batch % 3)) {
      let fourth = fetchAtIndex(randomOffset + 3);
      if (fourth[1] === planIndex) xBuffers.push(fourth[0]);
      else xBuffers.unshift(fetchAtIndex(randomOffset - 3)[0]);
      if (!(batch % 4)) {
        let fifth = fetchAtIndex(randomOffset + 4);
        if (fifth[1] === planIndex) xBuffers.push(fifth[0]);
        else xBuffers.unshift(fetchAtIndex(randomOffset - 4)[0]);
        set?.add(`${planIndex}:${planIndex}`);
      }
    }
  }

  secondHalf: {
    if (!(batch % 3) && !(batch % 4)) break secondHalf;
    let randomOffset = Math.floor(random() * count);
    let newPlanIndex = -1;
    let tries = 100;
    do {
      let randomOffset = Math.floor(random() * count);
      [, newPlanIndex] = fetchAtIndex(randomOffset);
      tries--;
    } while (
      tries &&
      (newPlanIndex === planIndex || set?.has(`${planIndex}:${newPlanIndex}`))
    );
    set?.add(`${planIndex}:${newPlanIndex}`);

    const first = fetchAtIndex(randomOffset);
    const push = batch % 5 ? "push" : "unshift";
    if (!(batch % 3)) {
      xBuffers[push](first[0]);
      break secondHalf;
    }

    isSamePlan = batch % 5 ? planIndex === first[1] : false;

    let second = fetchAtIndex(randomOffset + 1);
    if (second[1] === first[1]) {
      xBuffers[push](first[0]);
      xBuffers[push](second[0]);
    } else {
      xBuffers[push](fetchAtIndex(randomOffset - 1)[0]);
      xBuffers[push](first[0]);
    }
  }

  return [xBuffers, isSamePlan, planIndex] as const;
}
export interface IGenerateDatasetIteratorParams {
  fileHandler: number;
  batchSize: number;
  blockOffset: number;
  offset?: number;
  blockCount: number;
  batchCount?: number;
}
export function generateDatasetIterator({
  fileHandler,
  batchSize,
  blockOffset,
  blockCount,
  batchCount,
}: IGenerateDatasetIteratorParams) {
  const engine = tf.engine();
  const file = Buffer.alloc(
    blockCount * (IMAGE_BYTE_SIZE + PLAN_INDEX_BYTE_SIZE) + 8,
  );
  readSync(
    fileHandler,
    file,
    0,
    blockCount * (IMAGE_BYTE_SIZE + PLAN_INDEX_BYTE_SIZE) + 8,
    0,
  );
  return function* () {
    let lastTensors: tf.Tensor[] = [];
    const set = new Set<string>();
    for (
      let batch = 0;
      batch < (batchCount ?? blockCount / batchSize);
      batch++
    ) {
      const xs: Buffer[] = [];
      const ys: (0 | 1)[] = [];

      for (let i = 0; i < batchSize; i++) {
        const [buffers, isSamePlan] = pickImageSequenceF(
          file,
          batch,
          blockCount,
          blockOffset,
          set,
        );
        xs.push(joinBuffers(...buffers));
        ys.push(isSamePlan ? 0 : 1);
        // Mark the block as read
      }
      const tensors = {
        xs: createInputTensor(xs),
        ys: tf.tensor(ys.map((y) => [y])),
        batch,
      };
      yield tensors;
      if (lastTensors) lastTensors.forEach((t) => engine.disposeTensor(t));
      lastTensors = [tensors.xs, tensors.ys];
    }
    // if (lastTensors) lastTensors.forEach((t) => engine.disposeTensor(t));
  };
}
export function generateSequenceDatasetIterator({
  fileHandler,
  batchSize,
  blockOffset,
  blockCount,
  batchCount,
}: IGenerateDatasetIteratorParams) {
  const engine = tf.engine();
  return function* () {
    let lastTensors: tf.Tensor[] = [];
    const set = new Set<string>();
    for (
      let batch = 0;
      batch < (batchCount ?? blockCount / batchSize);
      batch++
    ) {
      const xs: Buffer[][] = [];
      const ys: (0 | 1)[] = [];

      for (let i = 0; i < batchSize; i++) {
        const [buffers, isSamePlan] = pickImageSequence(
          fileHandler,
          batch,
          blockCount,
          blockOffset,
          set,
        );

        xs.push(buffers);
        ys.push(isSamePlan ? 0 : 1);
        // Mark the block as read
      }
      const tensors = {
        xs: tf.tensor(xs.map((buf) => buf.map(reshape))),
        ys: tf.tensor(ys.map((y) => [y])),
        batch,
      };
      yield tensors;
      if (lastTensors) lastTensors.forEach((t) => engine.disposeTensor(t));
      lastTensors = [tensors.xs, tensors.ys];
    }
    // if (lastTensors) lastTensors.forEach((t) => engine.disposeTensor(t));
  };
}
export function generateMultiInputDatasetIterator({
  fileHandler,
  batchSize,
  blockOffset,
  blockCount,
  batchCount,
}: IGenerateDatasetIteratorParams) {
  const engine = tf.engine();
  return function* () {
    let lastTensors: [tf.Tensor[], tf.Tensor] | undefined = undefined;
    const set = new Set<string>();
    for (
      let batch = 0;
      batch < (batchCount ?? blockCount / batchSize);
      batch++
    ) {
      const xs: Buffer[][] = [[], [], [], [], []];
      const ys: (0 | 1)[] = [];

      for (let i = 0; i < batchSize; i++) {
        const [buffers, isSamePlan] = pickImageSequence(
          fileHandler,
          batch,
          blockCount,
          blockOffset,
          set,
        );
        buffers.forEach((buf, index) => xs[index].push(buf));

        // xs.push(joinBuffers(...xBuffers));
        ys.push(isSamePlan ? 0 : 1);
        // Mark the block as read
      }
      const tensors = {
        xs: xs.map((buffers) => createInputTensor(buffers)),
        ys: tf.tensor(ys.map((y) => [y])),
        batch,
      };
      yield tensors;
      if (lastTensors)
        lastTensors.forEach((t) =>
          Array.isArray(t)
            ? t.forEach((tt) => engine.disposeTensor(tt))
            : engine.disposeTensor(t),
        );
      lastTensors = [tensors.xs, tensors.ys];
    }
    // if (lastTensors) lastTensors.forEach((t) => engine.disposeTensor(t));
  };
}
export const ITERATORS = {
  default: generateDatasetIterator,
  multi: generateMultiInputDatasetIterator,
  sequence: generateSequenceDatasetIterator,
};
export function createInputTensor(data: Uint8Array[]) {
  return tf.tensor(data.map(reshape));
}
export function reshape(frame: Uint8Array) {
  return chunkArray(
    Array.from(frame).map((i) => [i]),
    IMAGE_SIZE.width,
  );
}

export function executeModel(
  model: tf.LayersModel,
  inputs: (Uint8Array | Buffer)[],
) {
  const response = model.predict(
    inputs.map((input) => createInputTensor([input])),
  ) as tf.Tensor<tf.Rank>;
  const [[res]] = response.arraySync() as [[number]];
  return res > 0.5;
}
export function executeModelWithTensors(
  model: tf.LayersModel,
  inputs: tf.Tensor[],
) {
  const response = model.predict(inputs) as tf.Tensor<tf.Rank>;
  const [[res]] = response.arraySync() as [[number]];
  return res > 0.5;
}
