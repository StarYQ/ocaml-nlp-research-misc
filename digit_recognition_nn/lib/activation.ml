(** Activation functions and their derivatives for neural networks *)

(** Sigmoid activation: σ(z) = 1 / (1 + e^(-z)) *)
let sigmoid z =
  Matrix.map (fun x -> 1.0 /. (1.0 +. exp (-.x))) z

(** Sigmoid derivative: σ'(z) = σ(z) * (1 - σ(z)) *)
let sigmoid_derivative z =
  let s = sigmoid z in
  Matrix.map2 (fun si _ -> si *. (1.0 -. si)) s z

(** ReLU activation: max(0, z) *)
let relu z =
  Matrix.map (fun x -> if x > 0.0 then x else 0.0) z

(** ReLU derivative: 1 if z > 0, else 0 *)
let relu_derivative z =
  Matrix.map (fun x -> if x > 0.0 then 1.0 else 0.0) z

(** Tanh activation: tanh(z) *)
let tanh z =
  Matrix.map Stdlib.tanh z

(** Tanh derivative: 1 - tanh²(z) *)
let tanh_derivative z =
  let t = tanh z in
  Matrix.map (fun ti -> 1.0 -. (ti *. ti)) t
