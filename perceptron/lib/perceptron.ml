module Perceptron = struct 
  type p = {
    inputs: float array;
    weights: float array;
    bias: float;
    activation: float -> float;
  }

  let init inputs weights bias activation = {
    inputs; 
    weights;
    bias; 
    activation
  }

  let output p = 
    let weighted_sum = 
      let sum = ref 0.0 in 
      for i=0 to (Array.length p.inputs)-1 do 
        sum := !sum +. (p.inputs.(i) *. p.weights.(i)) 
      done;
      !sum in p.activation (weighted_sum +. p.bias)
end;;