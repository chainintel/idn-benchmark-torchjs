'use strict';
var expect = require('chai').expect;
const { TorchBenchmark } = require('../dist/index.js');

describe('Benchmark TorchJS', () => {
  it('should run cpu', async () => {
    const benchmark = new TorchBenchmark('torch/cpu', 10);
    let score = await benchmark.getScore();
    console.log('cpu score', score);
  }).timeout(30000);

  it('should run gpu', async () => {
    const benchmark = new TorchBenchmark('torch/cuda', 10);
    let score = await benchmark.getScore();
    console.log('gpu score', score);
  }).timeout(30000);
});
