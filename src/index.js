import _ from 'lodash';
import * as d3 from "d3";

function slice() {
    switch (arguments.length) {
      case 0:
        return {
          start: 0,
          stop: null,
          step: 1
        };
      case 1:
        return {
          start: 0,
          stop: arguments[0],
          step: 1
        };
      case 2:
        return {
          start: arguments[0],
          stop: arguments[1],
          step: 1
        };
      case 3:
        return {
          start: arguments[0],
          stop: arguments[1],
          step: arguments[2]
        };
    }
  }

  function transformSize(size, slice) {
    let start = slice.start < 0 ? size + slice.start : slice.start;
    let stop;
    if (slice.step > 0) {
      if (!slice.stop) {
        stop = size;
      } else {
        stop = Math.min(slice.stop, size);
      }
    } else {
      if (!slice.stop) {
        stop = -1;
      } else {
        stop = Math.max(-1, slice.stop);
      }
    }
    return Math.floor((stop - start) / slice.step);
  }

  function bindSlices(slices, args) {
    let outerArgs = [];
    let k = 0;
    for (let [i, j] = [0, 0]; i < slices.length || j < args.length; ) {
      let s = slices[i];
      let a = args[j];
      if (typeof s == "number") {
        outerArgs.push(s);
        i++;
        k++;
      } else if (s === undefined) {
        outerArgs.push(a);
      } else {
        outerArgs.push(a * s.step + s.start);
        j++;
        i++;
        k++;
      }
    }
    return outerArgs;
  }

  function Indexer(shape, getter = (key) => key) {
    const getkey = (slices) => {
      slices = slices.map((s) => (Array.isArray(s) ? slice(...s) : s));

      if (slices.length < shape.length) {
        slices.push(
          ..._.range(shape.length - slices.length).map(() => slice())
        );
      }

      const newShape = [];
      for (let i = 0; i < shape.length; i++) {
        if (typeof slices[i] != "number") {
          newShape.push(transformSize(shape[i], slices[i]));
        }
      }

      if (newShape.length == 0) {
        return getter(slices);
      }

      return Indexer(newShape, (key) => getter(bindSlices(slices, key)));
    };
    return {
      shape: shape,
      getkey: getkey
    };
  }

  function determineShape(data) {
    const first = data.length;
    if (first === undefined) {
      return [];
    }

    return [first].concat(determineShape(data[0]));
  }

  function* indexiter(shape) {
    if (shape.length == 0) {
      yield [];
    }
    for (let i = 0; i < shape[0]; i++) {
      for (let index of indexiter(shape.slice(1))) {
        yield [i].concat(index);
      }
    }
  }

  class ndarray {
    constructor(_values, indexer) {
      this._values = _values;
      this.indexer = indexer ? indexer : Indexer(determineShape(_values));
      this.shape = this.indexer.shape;
    }
    getitem(key) {
      const keys = this.indexer.getkey(key);
      if (keys.getkey !== undefined) {
        return new ndarray(this._values, keys);
      }
      let item = this._values;
      for (key of keys.slice(0, keys.length - 1)) {
        if (item[key] === undefined) {
          item[key] = [];
        }
        item = item[key];
      }
      return item[keys[keys.length - 1]];
    }

    setitem(key, value) {
      const keys = this.indexer.getkey(key);
      if (keys.getkey !== undefined) {
        const slice = new ndarray(this._values, keys);
        for (let index of indexiter(slice.shape)) {
          slice.setitem(index, value.getitem(index));
        }
      } else {
        let item = this._values;
        for (key of keys.slice(0, keys.length - 1)) {
          item = item[key];
        }
        item[keys[keys.length - 1]] = value;
      }
    }

    *iter() {
      for (let i of _.range(this.shape[0])) {
        yield this.getitem([i]);
      }
    }

    asarray() {
      const out = [];
      for (let i of _.range(this.shape[0])) {
        if (this.shape.length == 1) {
          out.push(this.getitem([i]));
        } else {
          out.push(this.getitem([i]).asarray());
        }
      }
      return out;
    }
  }

  function empty(shape) {
    const tensor = new ndarray([], Indexer(shape));
    for (let i of indexiter(shape)) {
      tensor.getitem(i);
    }
    return tensor;
  }

  function fromfunction(shape, func) {
    const tensor = empty(shape);
    for (let i of indexiter(shape)) {
      tensor.setitem(i, func(...i));
    }
    return tensor;
  }

  function ones(shape) {
    const tensor = empty(shape);
    for (let i of indexiter(shape)) {
      tensor.setitem(i, 1);
    }
    return tensor;
  }

  function zeros(shape) {
    const tensor = empty(shape);
    for (let i of indexiter(shape)) {
      tensor.setitem(i, 0);
    }
    return tensor;
  }

  function setdefault(obj, key, value) {
    if (obj[key] === undefined) {
      obj[key] = value;
    }
    return obj[key];
  }

  function _einreduce(subscripts, operands, op, reducer, out) {
    // parse subscripts
    if (typeof subscripts == "string") {
      let [s1, s2] = subscripts.split("->");
      subscripts = [s1.split(",").map((s) => s.split("")), s2.split("")];
    }

    const [operandSubscripts, outputSubscripts] = subscripts;

    const inputSubscripts = _.union(
      operandSubscripts[0].concat(...operandSubscripts.slice(1))
    );
    const marginalSubscripts = _.difference(inputSubscripts, outputSubscripts);

    if (_.difference(outputSubscripts, inputSubscripts).length > 0) {
      throw "Found axes in output not occuring in input.";
    }

    let shape = {};

    // propagate shape and check consistent dimensions
    for (let [operand, subs] of _.zip(operands, operandSubscripts)) {
      for (let [sub, dim] of _.zip(subs, operand.shape)) {
        const existing_dim = setdefault(shape, sub, dim);
        if (existing_dim != dim) {
          throw "Found operands with matching subscripts and incompatible dimensions";
        }
      }
    }

    let outputShape = outputSubscripts.map((i) => shape[i]);
    let marginalShape = marginalSubscripts.map((i) => shape[i]);

    if (!out) {
      out = empty(outputShape);
    }

    // compute summation
    for (let outputIndices of indexiter(outputShape)) {
      let marginal = null;

      for (let marginalIndices of indexiter(marginalShape)) {
        let indices = Object.fromEntries(
          _.zip(
            outputSubscripts.concat(marginalSubscripts),
            outputIndices.concat(marginalIndices)
          )
        );

        const opArgs = [];

        for (let [subscripts, operand] of _.zip(operandSubscripts, operands)) {
          const operandIndices = subscripts.map(
            (subscript) => indices[subscript]
          );
          opArgs.push(operand.getitem(operandIndices));
        }
        if (marginal === null) {
          marginal = op(...opArgs);
        } else {
          marginal = reducer(marginal, op(...opArgs));
        }
      }
      if (outputShape.length == 0) {
        return marginal;
      }
      out.setitem(outputIndices, marginal);
    }

    return out;
  }

  function broadcastShapes(...shapes) {
    const out = [];
    for (let dims of _.zip(
      ...shapes.map((s) => s.slice().reverse())
    ).reverse()) {
      dims = _.uniq(dims.map((d) => _.defaultTo(d, 1)));
      if (dims.length > 2) {
        throw "Shapes cannot be broadcast together.";
      }

      out.push(Math.max(...dims));
    }
    return out;
  }

  function broadcastTo(arr, shape) {
    const extra_dims = shape.length - arr.shape.length;

    const keyfuncs = arr.shape.map((d, i) =>
      d == 1 ? (k) => 0 : (k) => k[extra_dims + i]
    );
    const _getkey = arr.indexer.getkey;

    return new ndarray(
      arr._values,
      Indexer(shape, (key) => {
        return _getkey(keyfuncs.map((f) => f(key)));
      })
    );
  }

  function broadcast(...arrs) {
    const shape = broadcastShapes(...arrs.map((a) => a.shape));
    return arrs.map((a) => broadcastTo(a, shape));
  }

  function tensorize(func) {
    // Takes a function that takes a n arguments and returns a scalar
    // and returns a function that takes a list of n tensors

    return (...tensors) => {
      tensors = tensors.map((t) => (typeof t == "number" ? array([t]) : t));
      const shape = broadcastShapes(...tensors.map((t) => t.shape));
      const out = empty(shape);
      tensors = tensors.map((t) => broadcastTo(t, shape));
      for (let indices of indexiter(shape)) {
        out.setitem(indices, func(...tensors.map((t) => t.getitem(indices))));
      }
      return out;
    };
  }

  function indices(shape) {
    // Returns a tensor of with all entries equal to their index
    const tensor = empty(shape.concat([shape.length]));
    for (let indices of indexiter(shape)) {
      for (let i of _.range(shape.length)) {
        tensor.setitem(indices.concat([i]), indices[i]);
      }
    }
    return tensor;
  }

  function array(data) {
    return new ndarray(data);
  }

  function einsum(subscripts, ...operands) {
    return _einreduce(
      subscripts,
      operands,
      (...args) => args.reduce((x, y) => x * y),
      (x, y) => x + y
    );
  }

  function matmul(a, b) {
    return einsum("ij,jk->ik", a, b);
  }

  function min(arr, axis) {
    if (!axis) {
      axis = _.range(arr.shape.length);
    } else if (!Array.isArray(axis)) {
      axis = [axis];
    }
    return _einreduce(
      [
        [_.range(arr.shape.length)],
        _.difference(_.range(arr.shape.length), axis)
      ],
      [arr],
      (x) => x,
      Math.min
    );
  }

  function max(arr, axis) {
    if (!axis) {
      axis = _.range(arr.shape.length);
    } else if (!Array.isArray(axis)) {
      axis = [axis];
    }
    return _einreduce(
      [
        [_.range(arr.shape.length)],
        _.difference(_.range(arr.shape.length), axis)
      ],
      [arr],
      (x) => x,
      Math.max
    );
  }

  function sum(arr, axis) {
    if (axis === undefined) {
      axis = _.range(arr.shape.length);
    }
    if (!Array.isArray(axis)) {
      axis = [axis];
    }
    return _einreduce(
      [
        [_.range(arr.shape.length)],
        _.difference(_.range(arr.shape.length), axis)
      ],
      [arr],
      (x) => x,
      (x, y) => x + y
    );
  }

  const np = {
    array: array,
    empty: empty,
    ones: ones,
    zeros: zeros,
    einsum: einsum,
    matmul: matmul,
    tensorize: tensorize,
    fromfunction: fromfunction,
    _einreduce: _einreduce,
    indexiter: indexiter,
    mul: tensorize((...X) => X.reduce((x, y) => x * y)),
    add: tensorize((...X) => X.reduce((x, y) => x + y)),
    div: tensorize((x, y) => x / y),
    sub: tensorize((x, y) => x - y),
    pow: tensorize((x, y) => x ** y),
    exp: tensorize((x) => Math.exp(x)),
    log: tensorize((x) => Math.log(x)),
    sin: tensorize((x) => Math.sin(x)),
    cos: tensorize((x) => Math.cos(x)),
    tan: tensorize((x) => Math.tan(x)),
    sinh: tensorize((x) => Math.sinh(x)),
    cosh: tensorize((x) => Math.cosh(x)),
    tanh: tensorize((x) => Math.tanh(x)),
    sqrt: tensorize((x) => Math.sqrt(x)),
    abs: tensorize((x) => Math.abs(x)),
    max: max,
    min: min,
    sum: sum,
    random: {
      normal: (shape) => fromfunction(shape, d3.randomNormal()),
      uniform: (shape) => fromfunction(shape, d3.randomUniform())
    },
    broadcastTo: broadcastTo,
    broadcastShapes: broadcastShapes,
    broadcast: broadcast
  };
 window.np = np;