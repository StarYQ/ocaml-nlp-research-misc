(* Define basic matrix operations *)
module Matrix = struct
  let create rows cols =
    Array.init rows (fun _ -> Array.make cols 0.0)

  let copy m =
    Array.map Array.copy m

  let rows m = Array.length m
  let cols m = Array.length m.(0)

  let map f m =
    Array.map (fun row -> Array.map f row) m

  let mapi f m =
    Array.mapi (fun i row -> Array.mapi (fun j x -> f i j x) row) m

  let add m1 m2 =
    Array.mapi (fun i row ->
        Array.mapi (fun j x -> x +. m2.(i).(j)) row
      ) m1

  let sub m1 m2 =
    Array.mapi (fun i row ->
        Array.mapi (fun j x -> x -. m2.(i).(j)) row
      ) m1

  let mul_elementwise m1 m2 =
    (* Element-wise multiplication *)
    Array.mapi (fun i row ->
        Array.mapi (fun j x -> x *. m2.(i).(j)) row
      ) m1

  let scalar_mul scalar m =
    map (fun x -> x *. scalar) m

  let dot m1 m2 =
    let r1 = rows m1
    and c1 = cols m1
    and r2 = rows m2
    and c2 = cols m2 in
    if c1 <> r2 then invalid_arg "Matrix.dot: dimension mismatch";
    let result = create r1 c2 in
    for i = 0 to r1 - 1 do
      for j = 0 to c2 - 1 do
        let sum = ref 0.0 in
        for k = 0 to c1 - 1 do
          sum := !sum +. m1.(i).(k) *. m2.(k).(j)
        done;
        result.(i).(j) <- !sum
      done
    done;
    result

  let transpose m =
    let r = rows m
    and c = cols m in
    let result = create c r in
    for i = 0 to r - 1 do
      for j = 0 to c - 1 do
        result.(j).(i) <- m.(i).(j)
      done
    done;
    result

  (* Convert a 1D array of biases to a matrix with the same number of rows as input. 
     - If input has shape (batch_size, in_features)
     - biases has shape out_features
     - We want to add biases row-wise to each sample in the batch.
  *)
  let add_bias input biases =
    let batch_size = rows input in
    let output = copy input in
    for i = 0 to batch_size - 1 do
      for j = 0 to (cols input) - 1 do
        output.(i).(j) <- output.(i).(j) +. biases.(j)
      done
    done;
    output
end

(* Define activation functions *)
module Activation = struct
  let sigmoid x = 1.0 /. (1.0 +. exp (-.x))
  let sigmoid_derivative x = x *. (1.0 -. x)

  let relu x = max 0.0 x
  let relu_derivative x = if x > 0.0 then 1.0 else 0.0

  let apply f = Matrix.map f
  let apply_derivative f = Matrix.map f
end

(* Define a cost function (MSE) and its derivative *)
module Cost = struct
  let mse outputs targets =
    let diff = Matrix.sub outputs targets in
    let diff_sq = Matrix.map (fun x -> x *. x) diff in
    let sum = ref 0.0 in
    Array.iter (fun row -> Array.iter (fun x -> sum := !sum +. x) row) diff_sq;
    !sum /. (float_of_int (Matrix.rows outputs * Matrix.cols outputs))

  let mse_derivative outputs targets =
    (* d/dx (1/2 * (output - target)^2 ) = (output - target) *)
    Matrix.sub outputs targets
end

(* Define the MLP module *)
module MLP = struct

  type t = {
    layers : int array;  (* Layer sizes, e.g. [|2; 3; 1|] *)
    weights : float array array array;  (* Each element: matrix (in_size x out_size) *)
    biases : float array array;         (* Each element: 1D array (out_size) *)
    activation : (float -> float) * (float -> float);  (* e.g. (sigmoid, sigmoid_derivative) *)
  }

  (* Initialize a network with random weights in [-0.05, 0.05], and zero biases *)
  let init layers (act, act_deriv) =
    let num_layers = Array.length layers in
    if num_layers < 2 then invalid_arg "MLP.init: need at least 2 layers";
    let weights = Array.init (num_layers - 1) (fun i ->
        let r = layers.(i) in
        let c = layers.(i+1) in
        let m = Matrix.create r c in
        Matrix.mapi (fun _ _ _ -> (Random.float 0.1) -. 0.05) m
      )
    in
    let biases = Array.init (num_layers - 1) (fun i ->
        Array.make layers.(i+1) 0.0
      )
    in
    { layers; weights; biases; activation = (act, act_deriv) }

  (* Forward pass. Returns (a_list, z_list):
     - a_list: array of activation matrices for each layer (including input)
     - z_list: array of 'z' matrices (pre-activation) for each layer (excluding input) *)
  let forward mlp input =
    let (act, _) = mlp.activation in
    let a_list = ref [|input|] in
    let z_list = ref [||] in

    (* For each layer i, we compute z = a_(i) * W_i + b_i
       then a_(i+1) = activation(z). *)
    Array.iteri (fun layer_i w ->
      let a_prev = (!a_list).(layer_i) in
      let z = Matrix.dot a_prev w in
      let z = Matrix.add_bias z mlp.biases.(layer_i) in
      let a_new = Activation.apply act z in
      (* Append z and a_new to lists *)
      z_list := Array.append !z_list [|z|];
      a_list := Array.append !a_list [|a_new|];
    ) mlp.weights;
    (!a_list, !z_list)

  (* Backpropagation: compute gradients (dW, dB) given a single batch (X, Y).
     - X has shape (batch_size, input_dim)
     - Y has shape (batch_size, output_dim)
  *)
  let backward mlp a_list z_list targets =
    let (_, act_derivative) = mlp.activation in
    let num_layers = Array.length mlp.layers in

    (* We'll store dW and dB in arrays of the same shape as mlp.weights and mlp.biases *)
    let dW = Array.map Matrix.copy mlp.weights in
    let dB = Array.map Array.copy mlp.biases in

    (* 1) Compute error at output layer:
       dZ_L = (a_L - Y) * act_derivative(z_L)  (element-wise multiplication)
       where L is the last layer index (num_layers - 1).
    *)
    let a_L = (a_list).(num_layers - 1) in
    let z_L = (z_list).(num_layers - 2) in   (* z_list has one less element than a_list *)
    let dZ_L =
      let diff = Cost.mse_derivative a_L targets in
      let d_act = Matrix.map act_derivative a_L in  (* derivative w.r.t. activated output *)
      Matrix.mul_elementwise diff d_act
    in

    (* We place this in a ref, so we can iterate backward. *)
    let dZ_prev = ref dZ_L in

    (* Start from the last layer (index = num_layers-2) and go backwards *)
    for i = (num_layers - 2) downto 0 do
      (* a_i has shape (batch_size, layers.(i)) *)
      let a_i = a_list.(i) in

      (* dW_i = a_i^T dot dZ_(i+1) / batch_size *)
      let batch_size = float_of_int (Matrix.rows a_i) in
      let dW_i = Matrix.dot (Matrix.transpose a_i) !dZ_prev in
      let dW_i = Matrix.scalar_mul (1.0 /. batch_size) dW_i in
      dW.(i) <- dW_i;

      (* dB_i = average of dZ_(i+1) across batch_size *)
      let dB_sum = Array.make (Array.length mlp.biases.(i)) 0.0 in
      (* sum dZ_prev across all batch rows *)
      Array.iter (fun row ->
        Array.iteri (fun j v ->
          dB_sum.(j) <- dB_sum.(j) +. v
        ) row
      ) !dZ_prev;
      for j = 0 to Array.length dB_sum - 1 do
        dB_sum.(j) <- dB_sum.(j) /. batch_size
      done;
      dB.(i) <- dB_sum;

      if i > 0 then
        (* dZ_(i) = dZ_(i+1) W_i^T * act_derivative(z_i) *)
        let w_i = mlp.weights.(i) in
        let dZ_i_part = Matrix.dot !dZ_prev (Matrix.transpose w_i) in
        let z_i = z_list.(i - 1) in
        (* We apply activation derivative to a_i or z_i, depending on your choice.
           If the activation derivative is in terms of the post-activation (like sigmoid),
           we can use a_i. If it's in terms of pre-activation (like ReLU done strictly),
           we might pass z_i. *)
        let d_act_i = Matrix.map act_derivative a_list.(i) in
        let dZ_i = Matrix.mul_elementwise dZ_i_part d_act_i in
        dZ_prev := dZ_i
    done;

    (dW, dB)

  (* Perform a single update (gradient descent step) on the MLP parameters *)
  let update mlp (dW, dB) learning_rate =
    Array.iteri (fun i w ->
      let w' = Matrix.sub w (Matrix.scalar_mul learning_rate dW.(i)) in
      mlp.weights.(i) <- w'
    ) mlp.weights;

    Array.iteri (fun i b ->
      let b' = Array.mapi (fun j bj -> bj -. learning_rate *. dB.(i).(j)) b in
      mlp.biases.(i) <- b'
    ) mlp.biases

  (* Train function: 
     - X: training inputs (batch_size, input_dim)
     - Y: training targets (batch_size, output_dim)
     - epochs: number of epochs
     - learning_rate: step size
     For brevity, we do a full-batch gradient descent. 
     You could split X, Y into mini-batches for actual practice.
  *)
  let train mlp ~inputs ~targets ~epochs ~learning_rate =
    for epoch = 1 to epochs do
      (* Forward pass *)
      let a_list, z_list = forward mlp inputs in
      (* Compute cost *)
      let cost_value = Cost.mse a_list.(Array.length mlp.layers - 1) targets in
      (* Backprop *)
      let dW, dB = backward mlp a_list z_list targets in
      (* Update parameters *)
      update mlp (dW, dB) learning_rate;
      if epoch mod 100 = 0 then
        Printf.printf "Epoch %d, Cost = %f\n%!" epoch cost_value;
    done

end

(* Usage example *)
let () =
  Random.self_init ();

  (* Suppose we have a small dataset, e.g., XOR problem: *)
  let inputs = [|
    [|0.0; 0.0|];
    [|0.0; 1.0|];
    [|1.0; 0.0|];
    [|1.0; 1.0|]
  |] in
  let targets = [|
    [|0.0|];
    [|1.0|];
    [|1.0|];
    [|0.0|]
  |] in

  (* Define the MLP structure: 2 -> 3 -> 1 *)
  let layers = [|2; 3; 1|] in
  let mlp = MLP.init layers (Activation.sigmoid, Activation.sigmoid_derivative) in

  (* Train the network *)
  let epochs = 10000 in
  let learning_rate = 0.1 in

  MLP.train mlp ~inputs ~targets ~epochs ~learning_rate;

  (* Test after training
  let test_input = [|[|0.0; 1.0|]|] in
  let a_list, _z_list = MLP.forward mlp test_input in
  let output = a_list.(Array.length layers - 1).(0).(0) in
  Printf.printf "Test [0 1] => %f\n" output;
  () *)
