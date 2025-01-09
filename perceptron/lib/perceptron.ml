module Perceptron = struct 
  type p = {
    weights: float array;
    bias: float;
    activation: float -> float;
  }

  let create weights bias activation = {
    weights;
    bias; 
    activation
  }

  let output p inputs = 
    let weighted_sum = 
      let sum = ref 0.0 in 
      for i=0 to (Array.length inputs)-1 do 
        sum := !sum +. (inputs.(i) *. p.weights.(i)) 
      done;
      !sum in p.activation (weighted_sum +. p.bias)
end;;