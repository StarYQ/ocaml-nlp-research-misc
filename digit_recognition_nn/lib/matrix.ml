(** Matrix operations for neural networks *)

type t = float array array

(** Create a matrix with given dimensions filled with a default value *)
let create rows cols default_value =
  Array.make_matrix rows cols default_value

(** Create a zero matrix *)
let zeros rows cols = create rows cols 0.0

(** Create a matrix filled with ones *)
let ones rows cols = create rows cols 1.0

(** Create a random matrix with values scaled by a factor *)
let random rows cols scale =
  Random.self_init ();
  Array.init rows (fun _ ->
    Array.init cols (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale))

(** Create a random matrix with normal distribution (approximation using Box-Muller) *)
let randn rows cols scale =
  Random.self_init ();
  Array.init rows (fun _ ->
    Array.init cols (fun _ ->
      let u1 = Random.float 1.0 in
      let u2 = Random.float 1.0 in
      let z = sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2) in
      z *. scale))

(** Get the shape of a matrix (rows, cols) *)
let shape mat =
  let rows = Array.length mat in
  if rows = 0 then (0, 0)
  else (rows, Array.length mat.(0))

(** Get element at position (i, j) *)
let get mat i j = mat.(i).(j)

(** Set element at position (i, j) *)
let set mat i j value = mat.(i).(j) <- value

(** Matrix multiplication: C = A × B *)
let dot a b =
  let rows_a, cols_a = shape a in
  let rows_b, cols_b = shape b in
  if cols_a <> rows_b then
    failwith (Printf.sprintf "Matrix dimensions incompatible for multiplication: (%d,%d) × (%d,%d)"
                rows_a cols_a rows_b cols_b);
  let result = zeros rows_a cols_b in
  for i = 0 to rows_a - 1 do
    for j = 0 to cols_b - 1 do
      let sum = ref 0.0 in
      for k = 0 to cols_a - 1 do
        sum := !sum +. (a.(i).(k) *. b.(k).(j))
      done;
      result.(i).(j) <- !sum
    done
  done;
  result

(** Element-wise addition *)
let add a b =
  let rows_a, cols_a = shape a in
  let rows_b, cols_b = shape b in
  if rows_a <> rows_b || cols_a <> cols_b then
    failwith "Matrix dimensions must match for addition";
  Array.mapi (fun i row ->
    Array.mapi (fun j _ -> a.(i).(j) +. b.(i).(j)) row
  ) a

(** Element-wise subtraction *)
let sub a b =
  let rows_a, cols_a = shape a in
  let rows_b, cols_b = shape b in
  if rows_a <> rows_b || cols_a <> cols_b then
    failwith "Matrix dimensions must match for subtraction";
  Array.mapi (fun i row ->
    Array.mapi (fun j _ -> a.(i).(j) -. b.(i).(j)) row
  ) a

(** Element-wise multiplication (Hadamard product) *)
let mult a b =
  let rows_a, cols_a = shape a in
  let rows_b, cols_b = shape b in
  if rows_a <> rows_b || cols_a <> cols_b then
    failwith "Matrix dimensions must match for element-wise multiplication";
  Array.mapi (fun i row ->
    Array.mapi (fun j _ -> a.(i).(j) *. b.(i).(j)) row
  ) a

(** Element-wise division *)
let div a b =
  let rows_a, cols_a = shape a in
  let rows_b, cols_b = shape b in
  if rows_a <> rows_b || cols_a <> cols_b then
    failwith "Matrix dimensions must match for element-wise division";
  Array.mapi (fun i row ->
    Array.mapi (fun j _ -> a.(i).(j) /. b.(i).(j)) row
  ) a

(** Scalar multiplication *)
let scalar_mult scalar mat =
  Array.map (fun row ->
    Array.map (fun x -> scalar *. x) row
  ) mat

(** Matrix transpose *)
let transpose mat =
  let rows, cols = shape mat in
  if rows = 0 || cols = 0 then mat
  else
    Array.init cols (fun j ->
      Array.init rows (fun i -> mat.(i).(j))
    )

(** Sum all elements in a matrix *)
let sum mat =
  Array.fold_left (fun acc row ->
    acc +. Array.fold_left (+.) 0.0 row
  ) 0.0 mat

(** Sum along an axis: 0 = sum columns (result shape: (1, cols)), 1 = sum rows (result shape: (rows, 1)) *)
let sum_axis axis mat =
  let rows, cols = shape mat in
  match axis with
  | 0 -> (* Sum columns: result is (1, cols) *)
      let result = zeros 1 cols in
      for j = 0 to cols - 1 do
        let col_sum = ref 0.0 in
        for i = 0 to rows - 1 do
          col_sum := !col_sum +. mat.(i).(j)
        done;
        result.(0).(j) <- !col_sum
      done;
      result
  | 1 -> (* Sum rows: result is (rows, 1) *)
      let result = zeros rows 1 in
      for i = 0 to rows - 1 do
        result.(i).(0) <- Array.fold_left (+.) 0.0 mat.(i)
      done;
      result
  | _ -> failwith "axis must be 0 or 1"

(** Apply a function element-wise *)
let map f mat =
  Array.map (fun row -> Array.map f row) mat

(** Apply a binary function element-wise *)
let map2 f a b =
  let rows_a, cols_a = shape a in
  let rows_b, cols_b = shape b in
  if rows_a <> rows_b || cols_a <> cols_b then
    failwith "Matrix dimensions must match for map2";
  Array.mapi (fun i row ->
    Array.mapi (fun j _ -> f a.(i).(j) b.(i).(j)) row
  ) a

(** Create matrix from list of lists *)
let of_lists lists =
  Array.of_list (List.map Array.of_list lists)

(** Convert matrix to list of lists *)
let to_lists mat =
  Array.to_list (Array.map Array.to_list mat)

(** Print matrix for debugging *)
let print mat =
  let rows, cols = shape mat in
  Printf.printf "Matrix (%d × %d):\n" rows cols;
  Array.iter (fun row ->
    Array.iter (fun x -> Printf.printf "%.4f " x) row;
    Printf.printf "\n"
  ) mat

(** Argmax along axis 0 (finds index of max value in each column) *)
let argmax_axis0 mat =
  let rows, cols = shape mat in
  let result = zeros 1 cols in
  for j = 0 to cols - 1 do
    let max_idx = ref 0 in
    let max_val = ref mat.(0).(j) in
    for i = 1 to rows - 1 do
      if mat.(i).(j) > !max_val then begin
        max_val := mat.(i).(j);
        max_idx := i
      end
    done;
    result.(0).(j) <- float_of_int !max_idx
  done;
  result

(** Broadcasting addition: add a (rows, 1) matrix to each column of a (rows, cols) matrix *)
let broadcast_add mat bias =
  let rows, cols = shape mat in
  let bias_rows, bias_cols = shape bias in
  if bias_rows <> rows || bias_cols <> 1 then
    failwith (Printf.sprintf "Cannot broadcast: mat is (%d,%d), bias is (%d,%d)"
                rows cols bias_rows bias_cols);
  Array.mapi (fun i row ->
    Array.map (fun x -> x +. bias.(i).(0)) row
  ) mat
