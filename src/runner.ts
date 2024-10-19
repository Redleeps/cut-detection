import { cp, mkdir, readdir, readFile } from "fs/promises";
import { rawBufferToPng, toRaw } from "./dataset";
import { executeModel, executeModelWithTensors, reshape } from "./ml";
import { askBoolean, joinBuffers } from "./utils";
import path from "path";
import * as tf from "@tensorflow/tfjs-node-gpu";

export async function runBase(
  directory: string,
  start: number,
  processor: (buffers: Buffer[], index: number) => boolean,
) {
  const dirFiles = (await readdir(directory, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".bmp"))
    .sort((a, b) => +a.name.slice(0, -4) - +b.name.slice(0, -4));
  console.log(dirFiles.length);
  const buffers: Buffer[] = [];
  let saveDir = path.join(directory, `run-${Date.now()}`);
  await mkdir(saveDir, { recursive: true }).catch(() => null);
  let planIndex = start;
  await mkdir(path.join(saveDir, (planIndex + "").padStart(8, "0"))).catch(
    () => null,
  );
  for (let i = 0; i < dirFiles.length; i++) {
    buffers.push(
      await toRaw(
        await readFile(path.join(dirFiles[i].parentPath, dirFiles[i].name)),
      ).toBuffer(),
    );
    if (buffers.length < 5) continue;
    process.stdout.write(`\rProcessing ${dirFiles[i - 2].name}`);
    await cp(
      path.join(dirFiles[i - 2].parentPath, dirFiles[i - 2].name),
      path.join(
        saveDir,
        (planIndex + "").padStart(8, "0"),
        dirFiles[i - 2].name,
      ),
    );
    if (processor(buffers, i)) {
      // await askBoolean(
      //   `I think frame ${dirFiles[i - 2].name} is the end of a plan`,
      // );
      await mkdir(path.join(saveDir, (planIndex + "").padStart(8, "0"))).catch(
        () => null,
      );
      planIndex++;
      await cp(
        path.join(dirFiles[i - 1].parentPath, dirFiles[i - 1].name),
        path.join(
          saveDir,
          (planIndex + "").padStart(8, "0"),
          dirFiles[i - 1].name,
        ),
      );
      buffers.shift();
    }
    buffers.shift();
  }
}
export async function run(model: tf.LayersModel, directory: string, start = 0) {
  return runBase(directory, start, (buffers) =>
    executeModel(model, [joinBuffers(...buffers)]),
  );
}
export async function runSequence(
  model: tf.LayersModel,
  directory: string,
  start = 0,
) {
  return runBase(directory, start, (buffers) =>
    executeModelWithTensors(model, [tf.tensor([buffers.map(reshape)])]),
  );
}
export async function runMultiInput(
  model: tf.LayersModel,
  directory: string,
  start = 0,
) {
  return runBase(directory, start, executeModel.bind(null, model));
}

export const RUNNERS = {
  default: run,
  multi: runMultiInput,
  sequence: runSequence,
};
