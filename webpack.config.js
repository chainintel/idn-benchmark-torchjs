// webpack.node.config.js

var webpack = require('webpack');
var path = require('path');
var libraryName = 'BenchmarkTorchJS';
var nodeExternals = require('webpack-node-externals');

var plugins = [];
var outputFile = 'index.js';

module.exports = {
  externals: [nodeExternals()],
  mode: 'production',
  optimization: {
    portableRecords: true,
    removeAvailableModules: true,
    mergeDuplicateChunks: true,
    occurrenceOrder: true,
    minimize: false,
    namedModules: true
  },
  watchOptions: {
    aggregateTimeout: 300,
    poll: 1000
  },
  module: {
    rules: [
      {
        loader: 'ts-loader',
        test: /\.tsx?$/,
        type: 'javascript/auto'
      },
      {
        loader: 'babel-loader',

        // Skip any files outside of your project's `src` directory
        include: [path.resolve(__dirname, 'src')],

        // Only run `.js` and `.jsx` files through Babel
        test: /\.jsx?$/,

        // Options to configure babel with
        query: {
          plugins: ['transform-runtime'],
          presets: ['es2015', 'stage-0']
        }
      },
      {
        test: /\.proto$/,
        use: ['json-loader', 'proto-loader6']
      }
    ]
  },
  output: {
    path: path.resolve(__dirname, './dist'),
    filename: outputFile,
    library: libraryName,
    libraryTarget: 'umd',
    umdNamedDefine: true
  },
  entry: [path.resolve(__dirname, './src/index.ts')],
  target: 'node',
  devtool: 'source-map',
  plugins: plugins,
  resolve: {
    extensions: ['.ts', '.js', '.json']
  }
};
