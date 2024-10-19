import { readdir } from "fs/promises";
import path from "path";

export function isNotEmptyArray<T>(arr: T[]): arr is [T, ...T[]] {
  return !!(arr.length && "0" in arr);
}
export function ratioSplitArray<T>(
  arr: readonly T[],
  maxTestLength = Infinity,
  testRatio = 0.2,
) {
  const testLength = Math.min(
    Math.round(arr.length * testRatio),
    maxTestLength,
  );
  return [arr.slice(0, testLength), arr.slice(testLength)];
}
export function chunkArray<T>(arr: T[], size: number): T[][] {
  const chunkedArr = [];
  for (let i = 0; i < arr.length; i += size) {
    chunkedArr.push(arr.slice(i, i + size));
  }
  return chunkedArr;
}
export function shuffleArray<T>(array: T[]) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

export function defer(fn: () => void) {
  const def = {
    isDefaultPrevented: false,
    [Symbol.dispose]: () => {
      if (def.isDefaultPrevented) return;
      fn();
    },
    preventDefault() {
      def.isDefaultPrevented = true;
    },
    exec() {
      def[Symbol.dispose]();
    },
  };
  return def;
}

export function entries<T extends Record<string, unknown>>(object: T) {
  return Object.entries(object) as [keyof T, T[keyof T]][];
}
export function fromEntries<T extends [string, unknown][]>(entries: T) {
  return Object.fromEntries(entries) as Record<T[number][0], T[number][1]>;
}
export function keys<T extends Record<string, unknown>>(object: T) {
  return Object.keys(object) as (keyof T)[];
}

export function counter<T>(fn: (iteration: number) => T, modulo = 1) {
  const _counter = {
    iteration: 0,
    increment(step = 1) {
      _counter.iteration += step;
      return _counter._updateCallback();
    },
    decrement(step = 1) {
      _counter.iteration -= step;
      return _counter._updateCallback();
    },
    reset() {
      _counter.decrement(_counter.iteration);
    },
    _updateCallback() {
      if (_counter.iteration % modulo) return;
      return fn(_counter.iteration);
    },
  };
  return _counter;
}
export function combinedIter<T>(...arrays: T[][]) {
  let currentArrayIndex = 0;
  let currentIndexInCurrentArray = 0;

  return <IterableIterator<T>>{
    [Symbol.iterator]() {
      return this;
    },
    next() {
      if (currentArrayIndex >= arrays.length) {
        return { done: true };
      }
      const currentArray = arrays[currentArrayIndex];
      if (currentIndexInCurrentArray >= currentArray.length) {
        currentIndexInCurrentArray = 0;
        currentArrayIndex++;
        return this.next();
      }
      const value = currentArray[currentIndexInCurrentArray];
      currentIndexInCurrentArray++;
      return { value, done: false };
    },
  };
}
export function panic(err: string): never {
  throw new Error(err);
}
export function mean(serie: number[]): number {
  if (serie.length === 0) {
    throw new Error("Cannot calculate mean of an empty array");
  }
  return sum(serie) / serie.length;
}
export function sum(serie: number[]): number {
  return serie.reduce((acc, val) => acc + val, 0);
}
export function movingAverage<T extends { x: number; y: number }>(
  data: T[],
  windowSize: number,
) {
  return data.map((point, index, array) => {
    const start = Math.max(0, index - windowSize + 1);
    const slice = array.slice(start, index + 1);
    const average = slice.reduce((sum, p) => sum + p.y, 0) / slice.length;
    return { x: point.x, y: average };
  });
}
export function expMovingAverage<T extends { x: number; y: number }>(
  data: T[],
  alpha: number,
) {
  const result: { x: number; y: number }[] = [];
  let previousAverage: number | null = null;

  data.forEach((point) => {
    if (previousAverage === null) {
      previousAverage = point.y; // Initialize with the first value
    } else {
      previousAverage = alpha * point.y + (1 - alpha) * previousAverage; // Exponential moving average formula
    }
    result.push({ x: point.x, y: previousAverage });
  });

  return result;
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

export function joinBuffers(...buffers: Buffer[]) {
  return buffers.reduce(
    (buf, cur) => Buffer.concat([buf, cur]),
    Buffer.alloc(0),
  );
}
export function askBoolean(question: string, duration?: number) {
  return new Promise<boolean>((resolve) => {
    process.stdout.write(`\n${question} (n for no)`);
    let timeout: NodeJS.Timeout;
    const listener = (data: Buffer) => {
      clearTimeout(timeout);
      resolve(data.toString().trim().toLowerCase() !== "n");
    };
    if (duration) {
      timeout = setTimeout(() => {
        process.stdout.write("\nTimeout, saving\n");
        process.stdin.off("data", listener);
        resolve(true);
      }, duration);
    }
    process.stdin.once("data", listener);
  });
}
export function seededRandom(seed: number) {
  let m = 0x80000000; // 2**31
  let a = 1103515245;
  let c = 12345;

  return function () {
    seed = (a * seed + c) % m;
    return seed / (m - 1);
  };
}
let _defaultRandom = seededRandom(Date.now());

export function random() {
  return _defaultRandom();
}
random.seed = function (seed: number) {
  _defaultRandom = seededRandom(seed);
};
