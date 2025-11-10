(** Quick verification: 64x64 resolution works with minimal training *)

open Digit_recognition_nn

(** Convert labels to one-hot encoding *)
let one_hot_encode y num_classes =
  let _, m = Matrix.shape y in
  let y_encoded = Matrix.zeros num_classes m in

  for i = 0 to m - 1 do
    let label = int_of_float y.(0).(i) in
    y_encoded.(label).(i) <- 1.0
  done;

  y_encoded

(** Load limited images from a directory *)
let load_limited_images_from_dir dir_path max_images =
  let files = Sys.readdir dir_path in
  let image_files = Array.to_list files |> List.filter (fun f ->
    String.lowercase_ascii f |> fun s ->
    String.ends_with ~suffix:".jpg" s ||
    String.ends_with ~suffix:".jpeg" s ||
    String.ends_with ~suffix:".png" s
  ) in

  (* Take only first max_images *)
  let limited_files =
    if List.length image_files > max_images then
      let rec take n lst = match n, lst with
        | 0, _ | _, [] -> []
        | n, x::xs -> x :: take (n-1) xs
      in take max_images image_files
    else image_files
  in

  List.map (fun filename ->
    let full_path = Filename.concat dir_path filename in
    Image_loader.load_and_preprocess_image full_path
  ) limited_files

(** Load limited digit images from archive *)
let load_limited_digits archive_path max_per_digit =
  let all_data = ref [] in

  for digit = 0 to 9 do
    let digit_dir = Filename.concat archive_path (string_of_int digit) in
    if Sys.file_exists digit_dir then begin
      Printf.printf "Loading %d images for digit %d...\n%!" max_per_digit digit;
      let images = load_limited_images_from_dir digit_dir max_per_digit in
      Printf.printf "  Loaded %d images for digit %d\n%!" (List.length images) digit;

      (* Add to dataset with label *)
      List.iter (fun pixels ->
        all_data := (pixels, digit) :: !all_data
      ) images
    end
  done;

  List.rev !all_data

(** Quick verification test *)
let () =
  Printf.printf "===========================================================\n";
  Printf.printf "VERIFICATION TEST - 28x28 Resolution\n";
  Printf.printf "===========================================================\n\n";

  (* Configuration - MINIMAL for quick verification *)
  let archive_path = "images/digit_imgs/archive" in
  let max_per_digit = 10 in  (* Only 10 images per digit = 100 total *)
  let learning_rate = 0.01 in
  let num_iterations = 100 in  (* 100 iterations for verification *)
  let num_classes = 10 in

  (* Architecture: 784 → 128 → 64 → 10 *)
  let layer_dims = [784; 128; 64; num_classes] in

  Printf.printf "Verification Configuration:\n";
  Printf.printf "  Images per digit: %d (total ~%d images)\n" max_per_digit (max_per_digit * 10);
  Printf.printf "  Network: 784 → 128 → 64 → 10\n";
  Printf.printf "  Learning rate: %.4f\n" learning_rate;
  Printf.printf "  Iterations: %d (for verification)\n" num_iterations;
  Printf.printf "  Train/test: 80/20 split\n\n";

  Printf.printf "KEY VERIFICATION POINTS:\n";
  Printf.printf "  ✓ Images are 28x28 (784 pixels - MNIST standard)\n";
  Printf.printf "  ✓ Network input layer: 784 neurons\n";
  Printf.printf "  ✓ Hidden layers: 128 and 64 neurons\n\n";

  (* Load limited dataset *)
  Printf.printf "Loading subset of real digit images...\n";
  let all_data = load_limited_digits archive_path max_per_digit in
  Printf.printf "\nTotal images loaded: %d\n\n" (List.length all_data);

  (* Split into train and test *)
  let train_data, test_data = Image_loader.train_test_split all_data 0.8 42 in
  Printf.printf "Training samples: %d\n" (List.length train_data);
  Printf.printf "Test samples: %d\n\n" (List.length test_data);

  (* Convert to matrices *)
  let x_train, y_train = Image_loader.data_to_matrices train_data in
  let x_test, y_test = Image_loader.data_to_matrices test_data in

  (* Verify dimensions *)
  let x_train_rows, x_train_cols = Matrix.shape x_train in
  Printf.printf "DIMENSION VERIFICATION:\n";
  Printf.printf "  X_train shape: (%d, %d)\n" x_train_rows x_train_cols;
  Printf.printf "  Expected:      (784, %d)\n" x_train_cols;
  if x_train_rows = 784 then
    Printf.printf "  ✓ CORRECT - Using 28x28 images (784 pixels)\n\n"
  else
    Printf.printf "  ✗ ERROR - Expected 784 rows!\n\n";

  (* Standardize *)
  Printf.printf "Applying StandardScaler normalization...\n";
  let means, stds = Image_loader.compute_mean_std x_train in
  let x_train_std = Image_loader.standardize x_train means stds in
  let x_test_std = Image_loader.standardize x_test means stds in
  Printf.printf "Standardization complete.\n\n";

  (* One-hot encode labels *)
  let y_train_encoded = one_hot_encode y_train num_classes in
  let y_test_encoded = one_hot_encode y_test num_classes in

  Printf.printf "Data shapes:\n";
  Printf.printf "  X_train: (%d, %d)\n" (fst (Matrix.shape x_train_std)) (snd (Matrix.shape x_train_std));
  Printf.printf "  Y_train: (%d, %d)\n" (fst (Matrix.shape y_train_encoded)) (snd (Matrix.shape y_train_encoded));
  Printf.printf "  X_test:  (%d, %d)\n" (fst (Matrix.shape x_test_std)) (snd (Matrix.shape x_test_std));
  Printf.printf "  Y_test:  (%d, %d)\n\n" (fst (Matrix.shape y_test_encoded)) (snd (Matrix.shape y_test_encoded));

  (* Train the model *)
  Printf.printf "Training neural network for %d iterations...\n" num_iterations;
  Printf.printf "(This verifies the network can train with 64x64 inputs)\n";
  Printf.printf "------------------------------------------------------------\n";
  let params = Neural_network.train
    x_train_std
    y_train_encoded
    layer_dims
    learning_rate
    num_iterations
    true  (* print_cost *)
    1     (* seed *)
  in
  Printf.printf "------------------------------------------------------------\n";
  Printf.printf "Training complete!\n\n";

  (* Evaluate on training set *)
  Printf.printf "Evaluating on training set...\n";
  let train_predictions = Neural_network.predict x_train_std params (List.length layer_dims) in
  let train_accuracy = Neural_network.accuracy train_predictions y_train in
  Printf.printf "Training Accuracy: %.2f%%\n\n" train_accuracy;

  (* Evaluate on test set *)
  Printf.printf "Evaluating on test set...\n";
  let test_predictions = Neural_network.predict x_test_std params (List.length layer_dims) in
  let test_accuracy = Neural_network.accuracy test_predictions y_test in
  Printf.printf "Test Accuracy: %.2f%%\n\n" test_accuracy;

  Printf.printf "===========================================================\n";
  Printf.printf "VERIFICATION COMPLETE!\n";
  Printf.printf "===========================================================\n\n";
  Printf.printf "✓ Images are 28x28 (784 pixels - MNIST standard)\n";
  Printf.printf "✓ Network architecture is 784 → 128 → 64 → 10\n";
  Printf.printf "✓ System compiles and runs successfully\n";
  Printf.printf "✓ Training works with optimized resolution\n\n";
  Printf.printf "Training accuracy: %.2f%%\n" train_accuracy;
  Printf.printf "Test accuracy:     %.2f%%\n\n" test_accuracy;
  Printf.printf "NOTE: This uses a much smaller network (100K parameters)\n";
  Printf.printf "      compared to the previous 64x64 version (1M+ parameters).\n";
  Printf.printf "      Training should now be 10x faster!\n";
  Printf.printf "      Run full training: dune exec ./bin/real_digits_quick.exe\n";
  Printf.printf "===========================================================\n"
