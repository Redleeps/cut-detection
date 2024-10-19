import { mkdir, open, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  defer,
  expMovingAverage,
  isNotEmptyArray,
  movingAverage,
} from "./utils";
import { cwd } from "node:process";
import sharp, { SharpOptions } from "sharp";
import formatDuration from "format-duration";
import {
  HEADER_BYTE_SIZE,
  IMAGE_BYTE_SIZE,
  IMAGE_SIZE,
  PLAN_INDEX_BYTE_SIZE,
} from "./constantes";
import { closeSync, openSync, readSync } from "node:fs";
//@ts-ignore
import visualiser from "visualiser";

export async function readImagesDir(directory: string) {
  const dataset: [string, ...string[]][] = [];
  const planDir = path.join(cwd(), "dataset", directory, "plans");
  const indexes = await readdir(planDir, { withFileTypes: true });
  for (const index of indexes) {
    if (index.isFile()) continue;
    let files = await getAllFilesInSubdirectories(
      path.join(planDir, index.name),
    );
    files = files.filter(
      (filename) => filename.endsWith(".bmp") || filename.endsWith(".png"),
    );
    if (isNotEmptyArray(files)) dataset.push(files);
  }
  return dataset;
}

export async function generateDatasetPlot(directory: string) {
  const dataset = await readImagesDir(directory);
  let inst = new visualiser({ width: 3000, height: 1200 });
  const frameWindow = 25; // * 30;
  const data = dataset
    .map(({ length }, x) => ({ x, y: length }))
    .reduce(
      (acc, cur) => {
        acc[0][acc[0].length - 1] += 1;
        while (cur.y >= frameWindow) {
          // acc[0][acc[0].length - 1] += 1;
          acc[0].push(0);
          cur.y -= frameWindow;
        }
        if (!cur.y) return acc;

        // acc[0][acc[0].length - 1] += 1;
        if (frameWindow - acc[1] > cur.y) {
          acc[1] += cur.y;
        } else {
          acc[0].push(0);
          acc[1] = cur.y - (frameWindow - acc[1]);
        }
        return acc;
      },
      [[0], 0] as [number[], number],
    )[0]
    .map((y, x) => ({ x, y }));
  inst.line({
    data: data,
    max_el: data.length,
    //@ts-ignore
    max_render: Math.max(...data.map((d) => d.y)) * 1.2,
    pointRadius: 0,
    lineColor: "deepskyblue",
    yGridCount: 20,
  });
  inst.line({
    data: movingAverage(data, 200),
    max_el: data.length,
    //@ts-ignore
    max_render: Math.max(...data.map((d) => d.y)) * 1.2,
    pointRadius: 0,
    lineColor: "orange",
    yGridCount: 0,
  });
  inst.line({
    data: movingAverage(data, data.length),
    max_el: data.length,
    //@ts-ignore
    max_render: Math.max(...data.map((d) => d.y)) * 1.2,
    pointRadius: 0,
    lineColor: "red",
    yGridCount: 0,
  });
  inst.line({
    data: expMovingAverage(data, 0.01),
    max_el: data.length,
    //@ts-ignore
    max_render: Math.max(...data.map((d) => d.y)) * 1.2,
    pointRadius: 0,
    lineColor: "green",
    yGridCount: 0,
  });
  const filename = `${directory}.plot.html`;
  await writeFile(filename, inst.visualise());
  console.log(`file://${cwd()}/${filename}`);
}

export async function getAllFilesInSubdirectories(directoryPath: string) {
  const files: string[] = [];

  try {
    const entries = await readdir(directoryPath, { withFileTypes: true });

    for (const entry of entries) {
      const filePath = path.join(directoryPath, entry.name);

      if (entry.isDirectory()) {
        files.push(...(await getAllFilesInSubdirectories(filePath)));
      } else {
        files.push(filePath);
      }
    }
  } catch (err) {
    console.error(`Error reading directory ${directoryPath}:`, err);
  }

  return files;
}

export function pngToRaw(png: Buffer) {
  return sharp(png).removeAlpha().toColourspace("b-w").raw();
}
export function toRaw(buff: Buffer) {
  let dv = new DataView(buff.buffer, buff.byteOffset, buff.byteLength);
  if (buff.toString("ascii", 0, 2) !== "BM") throw new Error("invalid bitmap");

  return sharp(
    new Uint8Array(
      buff.buffer,
      buff.byteOffset + dv.getUint32(10, true) /*pixel offset*/,
      dv.getUint32(34, true),
    ) /*data size*/
      .reverse()
      .slice(0),
    { raw: IMAGE_SIZE },
  )
    .flop()
    .removeAlpha()
    .toColourspace("b-w")
    .raw();
}

export async function reduceImagesToDataset<T>(
  names: string[],
  finalName?: string,
) {
  finalName ??= names[0];
  console.log("Creating dataset binary from images");
  const start = performance.now();
  const datasetDirectory = path.join(cwd(), "dataset", finalName);
  await using fileHandle = await open(datasetDirectory + ".bin", "w+");
  const writeStream = fileHandle.createWriteStream();

  const dataset = (await Promise.all(names.map(readImagesDir))).flat();
  const fullLength = dataset.reduce((a, c) => a + c.length, 0);
  const header = Buffer.alloc(HEADER_BYTE_SIZE);
  header.writeUint32LE(fullLength);
  writeStream.write(header);
  for (const planIndex in dataset) {
    const plan = dataset[planIndex];
    const indexBuffer = Buffer.alloc(4);
    indexBuffer.writeUint32LE(+planIndex);
    if (plan.length < 5) continue;
    for (const filePath of plan) {
      process.stdout.write(`${filePath.padEnd(60)}\r`);
      const imageBuffer = await (filePath.endsWith(".png") ? pngToRaw : toRaw)(
        await readFile(filePath),
      ).toBuffer();
      imageBuffer.writeUint32LE(0xffffffff);
      imageBuffer.writeUint32LE(0xffffffff, imageBuffer.byteLength - 4);
      writeStream.write(Buffer.concat([indexBuffer, imageBuffer]));
    }
  }
  await new Promise((r) => writeStream.end(r));
  const duration = performance.now() - start;
  console.log(`\nFinished in ${formatDuration(duration, { ms: true })}`);
  console.log("Written to dataset.bin");
}

export async function expandDatasetToImages<T>(name: string) {
  console.log("Creating dataset binary from images");
  const start = performance.now();
  const datasetDirectory = path.join(cwd(), "dataset", name);

  const { handler, length } = await openDatasetAsync(name);
  await using fileHandle = handler;

  const planIndexBuffer = Buffer.alloc(PLAN_INDEX_BYTE_SIZE);
  const imageBuffer = Buffer.alloc(IMAGE_BYTE_SIZE);

  for (let index = 0; index < length; index++) {
    process.stdout.write(
      `\r${(index + 1).toString().padStart(8)}/${length.toString().padEnd(8)}`,
    );
    const cursor =
      (IMAGE_BYTE_SIZE + PLAN_INDEX_BYTE_SIZE) * index + HEADER_BYTE_SIZE;
    await Promise.all([
      fileHandle.read(planIndexBuffer, 0, PLAN_INDEX_BYTE_SIZE, cursor),
      fileHandle.read(
        imageBuffer,
        0,
        IMAGE_BYTE_SIZE,
        cursor + PLAN_INDEX_BYTE_SIZE,
      ),
    ]);
    await mkdir(
      path.join(
        datasetDirectory,
        "plans",
        planIndexBuffer.readUint32LE(0).toString().padStart(9, "0"),
      ),
      { recursive: true },
    ).catch(() => {});
    await rawBufferToPng(imageBuffer).toFile(
      path.join(
        datasetDirectory,
        "plans",
        planIndexBuffer.readUint32LE(0).toString().padStart(9, "0"),
        `${index}.png`,
      ),
    );
  }

  const duration = performance.now() - start;
  console.log(`\nFinished in ${formatDuration(duration, { ms: true })}`);
}
export function rawBufferToPng(
  input: Parameters<typeof sharp>[0],
  raw: SharpOptions["raw"] = IMAGE_SIZE,
) {
  return sharp(input, { raw }).png();
}

export async function openDatasetAsync(name: string) {
  const datasetDirectory = path.join(cwd(), "dataset", name);
  const handler = await open(datasetDirectory + ".bin", "r");
  const header = Buffer.alloc(HEADER_BYTE_SIZE);
  await handler.read(header, 0, HEADER_BYTE_SIZE, 0);
  return {
    handler,
    length: header.readUint32LE(0),
    [Symbol.asyncDispose]() {
      return handler[Symbol.asyncDispose]();
    },
  };
}
export function openDatasetSync(name: string) {
  const datasetDirectory = path.join(cwd(), "dataset", name);
  const handler = openSync(datasetDirectory + ".bin", "r");
  const header = Buffer.alloc(HEADER_BYTE_SIZE);
  readSync(handler, header, 0, HEADER_BYTE_SIZE, 0);
  const length = header.readUint32LE(0);
  console.log(`${header.readUint32LE(0)} images in dataset`);
  return {
    handler,
    length,
    close: defer(() => closeSync(handler)),
    [Symbol.dispose]() {
      closeSync(handler);
    },
  };
}
