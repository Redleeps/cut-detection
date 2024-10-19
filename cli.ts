#!/usr/bin/env -S ts-node
import { hideBin } from "yargs/helpers";
import yargs from "yargs/yargs";
import {
  benchModel,
  compile,
  createModel,
  generateDatasetIterator,
  ITERATORS,
  loadModel,
  saveModel,
  trainModel,
} from "./src/ml";
import {
  expandDatasetToImages,
  generateDatasetPlot,
  reduceImagesToDataset,
} from "./src/dataset";
import { askBoolean, random } from "./src/utils";
import { run, RUNNERS } from "./src/runner";
// import { setTimeout } from "timers/promises";
random.seed(42);

async function cli() {
  await yargs(hideBin(process.argv))
    .scriptName("oh-fishing-bot")
    .command("model", "Run models", (yargs) => {
      return yargs
        .options({
          modelName: {
            type: "string",
            alias: ["model", "m"],
            required: true,
          },
          epochs: { type: "number", alias: "e", default: 100 },
          fitThreshold: {
            type: "number",
            alias: ["ft", "fit", "fitness"],
            default: 0.999,
          },
          learningRate: {
            type: "number",
            alias: ["l", "lr", "learning"],
            default: 0.001,
          },
          iterator: {
            alias: ["i", "it", "iterator"],
            type: "string",
            choices: ["default", "sequence", "multi"],
            required: true,
          },
          dataset: { type: "string", alias: "d", default: "dataset" },
          from: {
            type: "number",
            alias: ["f"],
            default: 0,
          },
        })
        .command({
          command: "run",
          describe: "Run the model on a directory",
          async handler({ modelName, dataset, iterator, from }) {
            const model = await loadModel(modelName);
            await (RUNNERS[iterator as keyof typeof RUNNERS] ?? run)(
              model,
              dataset,
              from,
            );
          },
        })
        .command({
          command: "bench",
          describe: "Assess model on the whole dataset",
          async handler({ modelName, dataset, iterator }) {
            const model = await loadModel(modelName);
            await benchModel(
              model,
              ITERATORS[iterator as keyof typeof ITERATORS] ??
                generateDatasetIterator,
              dataset,
            );
          },
        })
        .command({
          command: "train",
          describe: "Train model using dataset",
          async handler({
            modelName,
            epochs,
            dataset,
            fitThreshlod,
            learningRate,
            iterator,
          }) {
            const model = await loadModel(modelName);
            compile(model, learningRate);
            if (await askBoolean("Reset stored performances")) {
              model.setUserDefinedMetadata({});
            }
            await trainModel({
              model,
              epochs,
              dataset,
              fitThreshlod,
              iterator:
                ITERATORS[iterator as keyof typeof ITERATORS] ??
                generateDatasetIterator,
            });
          },
        })
        .demandCommand(1)
        .help().argv;
    })
    .command("dataset", "Manage datasets", (yargs) => {
      return yargs
        .options({
          dataset: { type: "array", alias: ["d"], default: ["dataset"] },
          output: { type: "string", alias: ["o"], default: "dataset" },
        })
        .command({
          command: "reduce",
          describe: "Write dataset directory to a binary",
          async handler({ dataset, output }) {
            await reduceImagesToDataset(dataset, output);
          },
        })
        .command({
          command: "plot",
          describe: "Write dataset directory to a binary",
          async handler({ dataset }) {
            console.log("dataset", dataset);
            await generateDatasetPlot(dataset[0]);
          },
        })
        .command({
          command: "expand",
          describe: "Expand a dataset",
          async handler({ dataset }) {
            await expandDatasetToImages(dataset[0]);
          },
        })
        .demandCommand(1)
        .help().argv;
    })
    .recommendCommands()
    .demandCommand(1)
    .help()
    .parse();
  process.stdin.destroy();
}

cli();
