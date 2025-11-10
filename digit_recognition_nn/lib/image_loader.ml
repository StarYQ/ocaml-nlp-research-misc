(** Image loading and preprocessing for digit recognition *)

open ImageLib_unix

(** Convert RGB image to grayscale using standard luminance formula:
    Gray = 0.299*R + 0.587*G + 0.114*B *)
let rgb_to_grayscale img =
  let open Image in
  let { width; height; max_val; pixels } = img in

  (* Create a new grayscale image *)
  let gray_img = create_grey ~max_val width height in

  (* Convert each pixel *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      match pixels with
      | RGB (r, g, b) ->
          let r_val = Pixmap.get r x y in
          let g_val = Pixmap.get g x y in
          let b_val = Pixmap.get b x y in
          (* Apply luminance formula *)
          let gray_val = int_of_float (
            0.299 *. float_of_int r_val +.
            0.587 *. float_of_int g_val +.
            0.114 *. float_of_int b_val
          ) in
          write_grey gray_img x y gray_val
      | RGBA (r, g, b, _) ->
          let r_val = Pixmap.get r x y in
          let g_val = Pixmap.get g x y in
          let b_val = Pixmap.get b x y in
          let gray_val = int_of_float (
            0.299 *. float_of_int r_val +.
            0.587 *. float_of_int g_val +.
            0.114 *. float_of_int b_val
          ) in
          write_grey gray_img x y gray_val
      | Grey _ | GreyA _ ->
          (* Already grayscale, just copy *)
          read_grey img x y (fun v -> write_grey gray_img x y v)
    done
  done;
  gray_img

(** Resize image to target dimensions using nearest neighbor interpolation *)
let resize_image img target_width target_height =
  let open Image in
  let { width; height; max_val; _ } = img in

  (* Create output image *)
  let resized = create_grey ~max_val target_width target_height in

  (* Nearest neighbor scaling *)
  let x_ratio = float_of_int width /. float_of_int target_width in
  let y_ratio = float_of_int height /. float_of_int target_height in

  for y = 0 to target_height - 1 do
    for x = 0 to target_width - 1 do
      let src_x = int_of_float (float_of_int x *. x_ratio) in
      let src_y = int_of_float (float_of_int y *. y_ratio) in
      (* Ensure we don't go out of bounds *)
      let src_x = min src_x (width - 1) in
      let src_y = min src_y (height - 1) in

      read_grey img src_x src_y (fun v ->
        write_grey resized x y v
      )
    done
  done;
  resized

(** Convert JPEG to PNG using imagemagick, then load
    This works around imagelib's buggy convert wrapper *)
let convert_jpeg_to_png_and_load filename =
  let is_jpeg =
    let lower = String.lowercase_ascii filename in
    String.ends_with ~suffix:".jpg" lower ||
    String.ends_with ~suffix:".jpeg" lower
  in

  if is_jpeg then begin
    (* Create temp PNG file *)
    let temp_png = Filename.temp_file "digit_img" ".png" in
    (* Use magick convert for ImageMagick v7 *)
    let cmd = Printf.sprintf "magick convert %s %s 2>/dev/null"
      (Filename.quote filename) (Filename.quote temp_png) in
    let exit_code = Sys.command cmd in
    if exit_code <> 0 then begin
      (* Cleanup and raise error *)
      if Sys.file_exists temp_png then Sys.remove temp_png;
      failwith (Printf.sprintf "Failed to convert %s to PNG" filename)
    end;
    (* Load the PNG *)
    let img = openfile temp_png in
    (* Clean up temp file *)
    Sys.remove temp_png;
    img
  end else
    (* Not a JPEG, load directly *)
    openfile filename

(** Load and preprocess a single image:
    1. Load from file
    2. Convert to grayscale
    3. Resize to 28x28
    4. Extract pixels as float array normalized to [0.0, 1.0] *)
let load_and_preprocess_image filename =
  (* Load image (with JPEG conversion workaround) *)
  let img = convert_jpeg_to_png_and_load filename in

  (* Convert to grayscale *)
  let gray_img = rgb_to_grayscale img in

  (* Resize to 28x28 *)
  let resized_img = resize_image gray_img 28 28 in

  (* Extract pixels and normalize *)
  let pixels = Array.make 784 0.0 in
  let max_val = float_of_int resized_img.max_val in

  for y = 0 to 27 do
    for x = 0 to 27 do
      let idx = y * 28 + x in
      Image.read_grey resized_img x y (fun v ->
        (* Normalize to [0.0, 1.0] *)
        pixels.(idx) <- float_of_int v /. max_val
      )
    done
  done;

  pixels

(** Load all images from a directory *)
let load_images_from_dir dir_path =
  let files = Sys.readdir dir_path in
  let image_files = Array.to_list files |> List.filter (fun f ->
    String.lowercase_ascii f |> fun s ->
    String.ends_with ~suffix:".jpg" s ||
    String.ends_with ~suffix:".jpeg" s ||
    String.ends_with ~suffix:".png" s
  ) in

  List.map (fun filename ->
    let full_path = Filename.concat dir_path filename in
    load_and_preprocess_image full_path
  ) image_files

(** Load all digit images from archive directory structure
    Returns list of (pixels, label) pairs *)
let load_all_digits archive_path =
  let all_data = ref [] in

  for digit = 0 to 9 do
    let digit_dir = Filename.concat archive_path (string_of_int digit) in
    if Sys.file_exists digit_dir then begin
      Printf.printf "Loading images for digit %d from %s...\n%!" digit digit_dir;
      let images = load_images_from_dir digit_dir in
      Printf.printf "  Loaded %d images for digit %d\n%!" (List.length images) digit;

      (* Add to dataset with label *)
      List.iter (fun pixels ->
        all_data := (pixels, digit) :: !all_data
      ) images
    end
  done;

  List.rev !all_data

(** Shuffle a list using Fisher-Yates algorithm *)
let shuffle_list lst seed =
  Random.init seed;
  let arr = Array.of_list lst in
  let n = Array.length arr in
  for i = n - 1 downto 1 do
    let j = Random.int (i + 1) in
    let temp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- temp
  done;
  Array.to_list arr

(** Split data into train and test sets *)
let train_test_split data train_ratio seed =
  let shuffled = shuffle_list data seed in
  let n = List.length shuffled in
  let n_train = int_of_float (float_of_int n *. train_ratio) in

  let rec split acc n lst =
    if n = 0 then (List.rev acc, lst)
    else match lst with
      | [] -> (List.rev acc, [])
      | x :: xs -> split (x :: acc) (n - 1) xs
  in
  split [] n_train shuffled

(** Convert list of (pixels, label) to matrix format
    X: (784, num_samples) matrix of features
    Y: (1, num_samples) matrix of labels *)
let data_to_matrices data =
  let num_samples = List.length data in
  let x_matrix = Array.make_matrix 784 num_samples 0.0 in
  let y_matrix = Array.make_matrix 1 num_samples 0.0 in

  List.iteri (fun i (pixels, label) ->
    (* Fill X matrix *)
    Array.iteri (fun j pixel_val ->
      x_matrix.(j).(i) <- pixel_val
    ) pixels;
    (* Fill Y matrix *)
    y_matrix.(0).(i) <- float_of_int label
  ) data;

  (x_matrix, y_matrix)

(** Compute mean and standard deviation for each feature *)
let compute_mean_std x_matrix =
  let n_features, n_samples = Matrix.shape x_matrix in
  let means = Array.make n_features 0.0 in
  let stds = Array.make n_features 0.0 in

  (* Compute means *)
  for i = 0 to n_features - 1 do
    let sum = ref 0.0 in
    for j = 0 to n_samples - 1 do
      sum := !sum +. x_matrix.(i).(j)
    done;
    means.(i) <- !sum /. float_of_int n_samples
  done;

  (* Compute standard deviations *)
  for i = 0 to n_features - 1 do
    let sum_sq = ref 0.0 in
    for j = 0 to n_samples - 1 do
      let diff = x_matrix.(i).(j) -. means.(i) in
      sum_sq := !sum_sq +. (diff *. diff)
    done;
    stds.(i) <- sqrt (!sum_sq /. float_of_int n_samples)
  done;

  (means, stds)

(** Standardize features using mean and std (Z-score normalization) *)
let standardize x_matrix means stds =
  let n_features, n_samples = Matrix.shape x_matrix in
  let x_standardized = Array.make_matrix n_features n_samples 0.0 in

  for i = 0 to n_features - 1 do
    for j = 0 to n_samples - 1 do
      (* Avoid division by zero *)
      let std = if stds.(i) < 1e-10 then 1.0 else stds.(i) in
      x_standardized.(i).(j) <- (x_matrix.(i).(j) -. means.(i)) /. std
    done
  done;

  x_standardized

(** Load and prepare digit dataset
    Returns (x_train, y_train, x_test, y_test) all as matrices *)
let load_dataset ?(train_ratio=0.8) ?(seed=42) ?(standardize_data=true) archive_path =
  Printf.printf "\n=== Loading Real Digit Images ===\n";
  Printf.printf "Archive path: %s\n\n" archive_path;

  (* Load all images *)
  let all_data = load_all_digits archive_path in
  Printf.printf "\nTotal images loaded: %d\n" (List.length all_data);

  (* Split into train and test *)
  Printf.printf "Splitting into train (%.0f%%) and test (%.0f%%)...\n"
    (train_ratio *. 100.0) ((1.0 -. train_ratio) *. 100.0);
  let train_data, test_data = train_test_split all_data train_ratio seed in
  Printf.printf "  Training samples: %d\n" (List.length train_data);
  Printf.printf "  Test samples: %d\n\n" (List.length test_data);

  (* Convert to matrices *)
  let x_train, y_train = data_to_matrices train_data in
  let x_test, y_test = data_to_matrices test_data in

  (* Standardize if requested *)
  let x_train_final, x_test_final =
    if standardize_data then begin
      Printf.printf "Applying StandardScaler normalization (mean=0, std=1)...\n";
      let means, stds = compute_mean_std x_train in
      let x_train_std = standardize x_train means stds in
      let x_test_std = standardize x_test means stds in
      Printf.printf "Standardization complete.\n\n";
      (x_train_std, x_test_std)
    end else begin
      Printf.printf "Using raw normalized pixel values [0.0, 1.0].\n\n";
      (x_train, x_test)
    end
  in

  (x_train_final, y_train, x_test_final, y_test)
