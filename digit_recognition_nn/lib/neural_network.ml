(** Deep Neural Network implementation from scratch *)

(** Activation function types *)
type activation_type = Sigmoid | ReLU

(** Parameters: weights and biases for each layer *)
type parameters = {
  weights: (string, Matrix.t) Hashtbl.t;
  biases: (string, Matrix.t) Hashtbl.t;
}

(** Cache for a single layer: stores values needed for backpropagation *)
type cache = {
  a_prev: Matrix.t;  (* Input to the layer *)
  w: Matrix.t;       (* Weights *)
  b: Matrix.t;       (* Biases *)
  z: Matrix.t;       (* Linear output before activation *)
}

(** Gradients for backpropagation *)
type gradients = {
  dw: (string, Matrix.t) Hashtbl.t;
  db: (string, Matrix.t) Hashtbl.t;
}

(** Initialize parameters for a deep neural network
    @param layer_dims List of layer dimensions [n_x, n_h1, n_h2, ..., n_y]
    @param seed Random seed for reproducibility
    @return Initialized parameters *)
let initialize_parameters layer_dims seed =
  Random.init seed;
  let l = List.length layer_dims in
  let weights = Hashtbl.create l in
  let biases = Hashtbl.create l in

  (* Initialize weights and biases for each layer *)
  for i = 1 to l - 1 do
    let n_prev = List.nth layer_dims (i - 1) in
    let n_curr = List.nth layer_dims i in

    (* W: (n_curr, n_prev), initialized with small random values *)
    let w = Matrix.randn n_curr n_prev 0.01 in
    Hashtbl.add weights ("W" ^ string_of_int i) w;

    (* b: (n_curr, 1), initialized with zeros *)
    let b = Matrix.zeros n_curr 1 in
    Hashtbl.add biases ("b" ^ string_of_int i) b;
  done;

  { weights; biases }

(** Linear forward: Z = W·A + b
    @param a Input activations from previous layer
    @param w Weights matrix
    @param b Bias vector
    @return (Z, cache) where Z is the linear output *)
let linear_forward a w b =
  let z_temp = Matrix.dot w a in
  let z = Matrix.broadcast_add z_temp b in
  let cache = { a_prev = a; w; b; z } in
  (z, cache)

(** Linear-activation forward: applies activation to linear output
    @param a_prev Activations from previous layer
    @param w Weights
    @param b Biases
    @param activation Activation function type
    @return (A, cache) where A is the activated output *)
let linear_activation_forward a_prev w b activation =
  let z, linear_cache = linear_forward a_prev w b in
  let a = match activation with
    | Sigmoid -> Activation.sigmoid z
    | ReLU -> Activation.relu z
  in
  (a, linear_cache)

(** Forward propagation for L-layer neural network
    @param x Input data
    @param parameters Network parameters
    @param num_layers Number of layers (including input and output)
    @return (AL, caches) where AL is final output, caches are stored for backprop *)
let l_model_forward x parameters num_layers =
  let caches = ref [] in
  let a = ref x in

  (* Forward through L-1 layers with ReLU *)
  for l = 1 to num_layers - 2 do
    let w = Hashtbl.find parameters.weights ("W" ^ string_of_int l) in
    let b = Hashtbl.find parameters.biases ("b" ^ string_of_int l) in
    let a_new, cache = linear_activation_forward !a w b ReLU in
    a := a_new;
    caches := cache :: !caches;
  done;

  (* Final layer with Sigmoid *)
  let l = num_layers - 1 in
  let w = Hashtbl.find parameters.weights ("W" ^ string_of_int l) in
  let b = Hashtbl.find parameters.biases ("b" ^ string_of_int l) in
  let al, cache = linear_activation_forward !a w b Sigmoid in
  caches := cache :: !caches;

  (al, List.rev !caches)

(** Compute cross-entropy cost
    @param al Final activations (predictions)
    @param y True labels
    @return Cost value *)
let compute_cost al y =
  let _, m_float = Matrix.shape y in
  let m = float_of_int m_float in

  (* Add small epsilon to avoid log(0) *)
  let epsilon = 1e-8 in
  let al_safe = Matrix.map (fun x ->
    if x < epsilon then epsilon
    else if x > 1.0 -. epsilon then 1.0 -. epsilon
    else x) al in

  (* cost = -(1/m) * sum(Y*log(AL) + (1-Y)*log(1-AL)) *)
  let y_log_al = Matrix.map2 (fun yi ali -> yi *. log ali) y al_safe in
  let one_y_log_one_al = Matrix.map2 (fun yi ali ->
    (1.0 -. yi) *. log (1.0 -. ali)) y al_safe in
  let sum_costs = Matrix.sum (Matrix.add y_log_al one_y_log_one_al) in
  let cost = -.(1.0 /. m) *. sum_costs in
  cost

(** Linear backward: compute gradients for linear part
    @param dz Gradient of cost with respect to Z
    @param cache Cache from forward pass
    @return (dA_prev, dW, db) gradients *)
let linear_backward dz cache =
  let { a_prev; w; b = _; z = _ } = cache in
  let _, m_float = Matrix.shape a_prev in
  let m = float_of_int m_float in

  (* dW = (1/m) * dZ · A_prev^T *)
  let dw = Matrix.scalar_mult (1.0 /. m) (Matrix.dot dz (Matrix.transpose a_prev)) in

  (* db = (1/m) * sum(dZ, axis=1, keepdims=True) *)
  let db = Matrix.scalar_mult (1.0 /. m) (Matrix.sum_axis 1 dz) in

  (* dA_prev = W^T · dZ *)
  let da_prev = Matrix.dot (Matrix.transpose w) dz in

  (da_prev, dw, db)

(** Activation backward: compute dZ from dA
    @param da Gradient with respect to activation
    @param cache Cache containing Z
    @param activation Activation function type
    @return dZ gradient *)
let activation_backward da cache activation =
  let z = cache.z in
  match activation with
  | ReLU ->
      let dz_activation = Activation.relu_derivative z in
      Matrix.mult da dz_activation
  | Sigmoid ->
      let dz_activation = Activation.sigmoid_derivative z in
      Matrix.mult da dz_activation

(** Linear-activation backward
    @param da Gradient with respect to activation
    @param cache Cache from forward pass
    @param activation Activation type
    @return (dA_prev, dW, db) *)
let linear_activation_backward da cache activation =
  let dz = activation_backward da cache activation in
  linear_backward dz cache

(** Backward propagation for L-layer network
    @param al Final activations
    @param y True labels
    @param caches Caches from forward pass
    @param num_layers Number of layers
    @return gradients *)
let l_model_backward al y caches num_layers =
  let l = num_layers - 1 in

  let dw_table = Hashtbl.create l in
  let db_table = Hashtbl.create l in

  (* Initialize backprop with derivative of cost *)
  (* dAL = -(Y/AL - (1-Y)/(1-AL)) *)
  let epsilon = 1e-8 in
  let al_safe = Matrix.map (fun x ->
    if x < epsilon then epsilon
    else if x > 1.0 -. epsilon then 1.0 -. epsilon
    else x) al in

  let dal = Matrix.map2 (fun yi ali ->
    -.(yi /. ali -. (1.0 -. yi) /. (1.0 -. ali))
  ) y al_safe in

  (* Backward through final layer (sigmoid) *)
  let current_cache = List.nth caches (l - 1) in
  let da_prev, dw, db = linear_activation_backward dal current_cache Sigmoid in
  Hashtbl.add dw_table ("dW" ^ string_of_int l) dw;
  Hashtbl.add db_table ("db" ^ string_of_int l) db;

  (* Backward through remaining layers (ReLU) *)
  let da_curr = ref da_prev in
  for layer = l - 1 downto 1 do
    let current_cache = List.nth caches (layer - 1) in
    let da_prev_temp, dw_temp, db_temp =
      linear_activation_backward !da_curr current_cache ReLU in
    da_curr := da_prev_temp;
    Hashtbl.add dw_table ("dW" ^ string_of_int layer) dw_temp;
    Hashtbl.add db_table ("db" ^ string_of_int layer) db_temp;
  done;

  { dw = dw_table; db = db_table }

(** Update parameters using gradient descent
    @param parameters Current parameters
    @param grads Computed gradients
    @param learning_rate Learning rate α
    @param num_layers Number of layers
    @return Updated parameters *)
let update_parameters parameters grads learning_rate num_layers =
  let new_weights = Hashtbl.create (num_layers - 1) in
  let new_biases = Hashtbl.create (num_layers - 1) in

  for l = 1 to num_layers - 1 do
    let w_key = "W" ^ string_of_int l in
    let b_key = "b" ^ string_of_int l in
    let dw_key = "dW" ^ string_of_int l in
    let db_key = "db" ^ string_of_int l in

    let w = Hashtbl.find parameters.weights w_key in
    let b = Hashtbl.find parameters.biases b_key in
    let dw = Hashtbl.find grads.dw dw_key in
    let db = Hashtbl.find grads.db db_key in

    (* W := W - α * dW *)
    let new_w = Matrix.sub w (Matrix.scalar_mult learning_rate dw) in
    (* b := b - α * db *)
    let new_b = Matrix.sub b (Matrix.scalar_mult learning_rate db) in

    Hashtbl.add new_weights w_key new_w;
    Hashtbl.add new_biases b_key new_b;
  done;

  { weights = new_weights; biases = new_biases }

(** Train the neural network
    @param x Training data
    @param y Training labels
    @param layer_dims Layer dimensions
    @param learning_rate Learning rate
    @param num_iterations Number of training iterations
    @param print_cost Whether to print cost during training
    @param seed Random seed
    @return Trained parameters *)
let train x y layer_dims learning_rate num_iterations print_cost seed =
  let num_layers = List.length layer_dims in
  let parameters = ref (initialize_parameters layer_dims seed) in

  for i = 0 to num_iterations - 1 do
    (* Forward propagation *)
    let al, caches = l_model_forward x !parameters num_layers in

    (* Compute cost *)
    let cost = compute_cost al y in

    (* Backward propagation *)
    let grads = l_model_backward al y caches num_layers in

    (* Update parameters *)
    parameters := update_parameters !parameters grads learning_rate num_layers;

    (* Print cost *)
    if print_cost && i mod 1000 = 0 then
      Printf.printf "Cost after iteration %d: %f\n%!" i cost;
  done;

  !parameters

(** Make predictions using trained model
    @param x Input data
    @param parameters Trained parameters
    @param num_layers Number of layers
    @return Predictions (class indices) *)
let predict x parameters num_layers =
  let al, _ = l_model_forward x parameters num_layers in
  Matrix.argmax_axis0 al

(** Compute accuracy
    @param predictions Predicted labels
    @param y True labels
    @return Accuracy as a percentage *)
let accuracy predictions y =
  let _, m_float = Matrix.shape y in
  let m = float_of_int m_float in
  let correct = ref 0.0 in

  for i = 0 to int_of_float m - 1 do
    if predictions.(0).(i) = y.(0).(i) then
      correct := !correct +. 1.0
  done;

  (!correct /. m) *. 100.0
