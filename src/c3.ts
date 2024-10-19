import * as tf from "@tensorflow/tfjs";

export class C3 extends tf.layers.Layer {
  filters: number;
  static readonly className = C3.name;
  constructor(filters: number) {
    super({});
    this.filters = filters;
  }
  convBlock(kernelSize: number, strides: number) {
    return tf.layers.conv2d({
      filters: this.filters,
      kernelSize,
      strides,
      activation: "relu",
      padding: "same",
    });
  }

  override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    const input = [inputs].flat()[0];
    let left = this.convBlock(1, 1).apply(input) as tf.SymbolicTensor;
    let right = this.convBlock(1, 1).apply(input) as tf.SymbolicTensor;
    right = this.convBlock(3, 1).apply(right) as tf.SymbolicTensor;
    right = this.convBlock(3, 1).apply(right) as tf.SymbolicTensor;

    let sum = tf.layers.add().apply([left, right]) as tf.SymbolicTensor;

    return this.convBlock(1, 1).apply(sum) as tf.Tensor;
  }
}
tf.serialization.registerClass(C3);
