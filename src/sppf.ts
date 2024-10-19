import * as tf from "@tensorflow/tfjs";
import { panic } from "./utils";

export type DataFormat = "channelsFirst" | "channelsLast";

export interface SpatialPyramidPoolingArgs {
  bins: (number | [number, number])[];
  dataFormat?: DataFormat;
}

/**
  Performs Spatial Pyramid Pooling.

    See [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf).

    Spatial Pyramid Pooling generates a fixed-length representation
    regardless of input size/scale. It is typically used before a layer
    that requires a constant input shape, for example before a Dense Layer.

    Args:
    bins: Either a collection of integers or a collection of collections of 2 integers.
        Each element in the inner collection must contain 2 integers, (pooled_rows, pooled_cols)
        For example, providing [1, 3, 5] or [[1, 1], [3, 3], [5, 5]] preforms pooling
        using three different pooling layers, having outputs with dimensions 1x1, 3x3 and 5x5 respectively.
        These are flattened along height and width to give an output of shape
        [batch_size, (1 + 9 + 25), channels] = [batch_size, 35, channels].
    dataFormat: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
    - If `dataFormat='channels_last'`:
        4D tensor with shape `(batch_size, height, width, channels)`.
    - If `dataFormat='channels_first'`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:
    The output is the pooled image, flattened across its height and width
    - If `dataFormat='channels_last'`:
        3D tensor with shape `(batch_size, num_bins, channels)`.
    - If `dataFormat='channels_first'`:
        3D tensor with shape `(batch_size, channels, num_bins)`.
*/
export class SpatialPyramidPooling extends tf.layers.Layer {
  bins: [number, number][];
  dataFormat: DataFormat;
  poolLayers: AdaptiveAveragePooling2D[];
  static readonly className = SpatialPyramidPooling.name;
  constructor({
    bins,
    dataFormat = "channelsLast",
    ...layerArgs
  }: SpatialPyramidPoolingArgs) {
    super(layerArgs);

    this.bins = bins.map((b) => (typeof b === "number" ? [b, b] : b));
    this.dataFormat = dataFormat;
    this.poolLayers = this.bins.map(
      (bin) => new AdaptiveAveragePooling2D(bin, this.dataFormat),
    );
  }
  override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const input = [inputs].flat()[0];
      const dynamicInputShape = input.shape as [number, number, number, number];
      let outputs = <tf.Tensor[]>[];
      let index = 0;

      if (this.dataFormat === "channelsLast") {
        for (const bin of this.bins) {
          const heightOverflow =
            bin[0] < dynamicInputShape[1] ? dynamicInputShape[1] % bin[0] : 0;
          const widthOverflow =
            bin[1] < dynamicInputShape[2] ? dynamicInputShape[2] % bin[1] : 0;
          const newInputHeight = dynamicInputShape[1] - heightOverflow;
          const newInputWidth = dynamicInputShape[2] - widthOverflow;
          const newInput = input.slice(
            [0, 0, 0, 0],
            [
              dynamicInputShape[0],
              newInputHeight,
              newInputWidth,
              dynamicInputShape[3],
            ],
          );
          const output = this.poolLayers[index].apply(newInput) as tf.Tensor;
          const reshapedOutput = tf.reshape(
            output,
            // @ts-ignore
            this.computeOutputShape(dynamicInputShape),
          );
          outputs.push(reshapedOutput);
        }
        index += 1;
      } else {
        panic("Not implemented");
        // const newInputs = inputs.map((input) => {
        //   const heightOverflow = dynamicInputShape[2]! % this.bins[index][0];
        //   const widthOverflow = dynamicInputShape[3]! % this.bins[index][1];
        //   const newInputHeight = dynamicInputShape[2]! - heightOverflow;
        //   const newInputWidth = dynamicInputShape[3]! - widthOverflow;
        //   return input.slice(0, 0, newInputHeight, newInputWidth);
        // });
        // outputs.push(...newInputs.map((newInp) => {
        //   const output = this.poolLayers[index].apply(newInp);
        //   const reshapedOutput = tf.reshape(output, [dynamicInputShape[0], inputs.shape[1], this.bins[index][0] * this.bins[index][1]]);
        //   return reshapedOutput;
        // }));
        // index += 1;
        // return tf.concat(outputs, 2);
      }
      return tf.concat(outputs);
    });
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    let pooledShape = 0;
    for (const bin of this.bins) {
      pooledShape += bin[0] * bin[1];
    }

    if (this.dataFormat === "channelsLast") {
      return [inputShape[0], pooledShape, inputShape[3]];
    } else {
      return [inputShape[0], inputShape[1], pooledShape];
    }
  }

  getConfig(): tf.serialization.ConfigDict {
    return { bins: this.bins, dataFormat: this.dataFormat };
  }
}

export class AdaptivePooling2D extends tf.layers.Layer {
  private reduceFunction: (x: tf.Tensor, axis: number[]) => tf.Tensor;
  private outputSize: [number, number];
  private dataFormat: DataFormat;
  static readonly className = AdaptivePooling2D.name;
  constructor(
    reduceFunction: (x: tf.Tensor, axis: number[]) => tf.Tensor,
    outputSize: [number, number],
    dataFormat: DataFormat = "channelsLast",
  ) {
    super();
    this.reduceFunction = reduceFunction;
    this.outputSize = outputSize;
    this.dataFormat = dataFormat;
  }

  call(inputs: tf.Tensor): tf.Tensor {
    const [hBins, wBins] = this.outputSize;

    if (this.dataFormat === "channelsLast") {
      const splitCols = tf.split(inputs, hBins, 1);
      const splitColsStacked = tf.stack(splitCols, 1);
      const splitRows = tf.split(splitColsStacked, wBins, 3);
      const splitRowsStacked = tf.stack(splitRows, 3);
      return this.reduceFunction(splitRowsStacked, [2, 4]);
    } else {
      const splitCols = tf.split(inputs, hBins, 2);
      const splitColsStacked = tf.stack(splitCols, 2);
      const splitRows = tf.split(splitColsStacked, wBins, 4);
      const splitRowsStacked = tf.stack(splitRows, 4);
      return this.reduceFunction(splitRowsStacked, [3, 5]);
    }
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    const [batchSize, height, width, channels] = inputShape;

    if (this.dataFormat === "channelsLast") {
      return [batchSize, this.outputSize[0], this.outputSize[1], channels];
    } else {
      return [batchSize, channels, this.outputSize[0], this.outputSize[1]];
    }
  }

  getConfig(): tf.serialization.ConfigDict {
    return {
      outputSize: this.outputSize,
      dataFormat: this.dataFormat,
    };
  }
}

export class AdaptiveAveragePooling2D extends AdaptivePooling2D {
  static readonly className = AdaptiveAveragePooling2D.name;
  constructor(bin: [number, number], dataFormat: DataFormat) {
    super(tf.mean, bin, dataFormat);
  }
}
tf.serialization.registerClass(AdaptivePooling2D);
tf.serialization.registerClass(AdaptiveAveragePooling2D);
tf.serialization.registerClass(SpatialPyramidPooling);
