### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ ba491ad9-0c09-44a3-ab98-21919da7c62e
begin
	# activate the project in the current directory
	using Pkg
	Pkg.activate("")

	# add our own model and augmentation modules
	include("./src/model_backend.jl")
	include("./src/augmentation_backend.jl")
	using .LeNet5, .Augmentation

	# used for the ToC
	using PlutoUI 

	# used to display the sample images
	using GLMakie

	# only needed for onecold to prep some data
	using Flux
	import Flux: onecold

	# used for saving and loading the model
	using BSON
	import BSON: @save, @load
	
	PlutoUI.TableOfContents()
end

# ╔═╡ d1cbf365-b3b1-494f-8158-f7c3b33a9fd3
md"# Digit Recognition - Data Augmentation"

# ╔═╡ a8cfdd5a-c5ef-4ef0-946c-d9594724acc3
md"## 1 - Introduction"

# ╔═╡ 9dc2706a-7b71-4576-bb04-68c7aacbb9ad
HTML("<p style='font-size:18px;'>
Digit recognition enables machines to interpret and process handwritten or printed numerical data, bridging human input with digital systems.
</br></br>
Data augmentation helps in training for digit recognition by artificially expanding the training dataset through transformations like rotation or noise addition, allowing the model to generalize better for real-world usage.
</p>")

# ╔═╡ dc6459b8-92a2-45c1-8378-5089def92f15
md"## 2 - Motivation"

# ╔═╡ 4b142462-d76d-4948-8de0-b32b65f4f0b7
HTML("<p style='font-size:18px;'>
In this project, we train and test models implemented with a convolutional neural network (CNN) architecture based on 'LeNet-5'.
</br></br>
We first train a model on most<sup>1</sup> of the full MNIST dataset (60.000 images) and test it on the MNIST test set (10.000 images). This represents our 'base-case', and its values serve as guide-values.
</br></br>
In real-world applications, usable data is often limited, and research time and computation power are constrained. Therefore our goal is to apply various augmentation methods to achieve comparable results using only fractions of the training data in a fraction of the time and processing power.
</br></br>
<span style='font-size:14px;'><sup>1</sup>More on this in section 4</span>
</p>")

# ╔═╡ 51c66468-da97-42a0-b594-e1a531ce66a2
md"## 3 - Data Augmentation Methods"

# ╔═╡ 132df693-22dc-4757-8f21-5421317f836b
begin
	# get a small (10 each) data set to visualize the augmentation
	data_small = LeNet5.getData_train(; amounts=fill(1,10))

	# apply all kinds of augmentation
	(data_rotate_x, data_rotate_y)= Augmentation.apply_augmentation_rotate(data_small[1], data_small[2])
	(data_noise_x, data_noise_y)= Augmentation.apply_augmentation_noise(data_small[1], data_small[2])
	(data_flip_x, data_flip_y)= Augmentation.apply_augmentation_flip(data_small[1], data_small[2])
	(data_all_x, data_all_y)= Augmentation.apply_augmentation_full(data_small[1], data_small[2])

	# create all figures
	fig_aug_rot = LeNet5.makeFigurePluto_Images(1400,175,data_rotate_x, data_rotate_y)
	fig_aug_noise = LeNet5.makeFigurePluto_Images(1400,175,data_noise_x, data_noise_y)
	fig_aug_flip = LeNet5.makeFigurePluto_Images(1400,175,data_flip_x, data_flip_y)
	fig_aug_full = LeNet5.makeFigurePluto_Images(1400,175,data_all_x, data_all_y)
	fig_no_aug = LeNet5.makeFigurePluto_Images(1400,175,data_small[1],data_small[2])
	nothing
end

# ╔═╡ f5762b3f-37af-45f0-aecc-d7785b58984b
md"### 3.1 - No Augmentation"

# ╔═╡ bd8f9222-1c39-45a0-b29f-c9e2b7018e79
fig_no_aug

# ╔═╡ 4eab70fe-25be-44a3-956c-899e5712a790
md"### 3.2 - Rotation"

# ╔═╡ 9ed2b7e1-9da8-49d6-a463-b23d452d1ee7
fig_aug_rot

# ╔═╡ d112f3f9-700a-4d40-882a-db90c3ed7706
md"### 3.3 - Noise"

# ╔═╡ b5420a56-02c0-4f9a-9a80-457d039dd1f3
fig_aug_noise

# ╔═╡ 1b55e14d-465d-4458-9d68-5d4da206a226
md"### 3.4 - Flip (Mirror)"

# ╔═╡ 085188be-e85f-476c-9aff-fe87a4dcf423
fig_aug_flip

# ╔═╡ ae063e85-0ed0-40b1-b84c-82e3e6de0d1f
md"### 3.5 - Full Augmentation w/o Flip"

# ╔═╡ 34189beb-75e1-4bec-acf8-d726515e80e6
fig_aug_full

# ╔═╡ 3152263a-e1b5-43c8-b957-9f50c66b2fc4
md"## 4 - Trained Models"

# ╔═╡ efc1f254-e5b0-4cc6-94f3-13ca9c77aa9c
HTML("<p style='font-size:18px;'>
We train the base-case model, hereafter referred to as the 'fully trained model', on 5421 images of each digit (54210 images in total) to be able to reliably compare the accuracy per label in testing.
</br></br>
We train the models with fractional input on 542 images per digit (10% of the fully trained model) using the same training function, modifying only the augmentation methods to analyze their impact.
</p>")

# ╔═╡ 22cddec6-24bc-4afd-9efb-a428880355ea
begin
	epochs_full = 20
	batchsize_full = 32
	lambda_full = 0
	eta_full = 3e-4
	set_full = 5421
	
	epochs_frac = 20
	batchsize_frac = 16
	lambda_frac = 1e-3
	eta_frac = 3e-3
	set_frac = 270
	nothing
end

# ╔═╡ 146b2db4-3834-453d-887e-e70cc7f9de20
md"""
|               | Epochs | Batchsize | λ | η | Training Set Size |
|---------------|--------|-----------|---|---|-------------------|
| Fully Trained | $(epochs_full) | $(batchsize_full) | $(lambda_full) | $(eta_full) | $(set_full * 10) |
| Fractional    | $(epochs_frac) | $(batchsize_frac) | $(lambda_frac) | $(eta_frac) | $(set_frac * 10) |
"""

# ╔═╡ 33da904c-a964-4305-a5cd-eff2fe6543c9
begin
	### Initilize all models
	model_NoAug = LeNet5.createModel() 
	model_Rotation = LeNet5.createModel() 
	model_Noise = LeNet5.createModel() 
	model_Flip = LeNet5.createModel()
	model_FullAug = LeNet5.createModel()
	model_full = LeNet5.createModel()

	
	### get training data
	data_full = LeNet5.getData_train(; amounts=fill(set_full,10))
	data_part = LeNet5.getData_train(; amounts=fill(set_frac,10))

	
	### create the model dict
	aug_fun::Function = (a, b) -> (a, b)
	
	mutable struct model_struct
		aug_function::Function
	
		model::Flux.Chain{Tuple{Flux.Conv{2, 4, typeof(NNlib.relu), Array{Float32, 4}, Vector{Float32}}, Flux.MaxPool{2, 4}, Flux.Conv{2, 4, typeof(NNlib.relu), Array{Float32, 4}, Vector{Float32}}, Flux.MaxPool{2, 4}, typeof(Flux.flatten), Flux.Dense{typeof(NNlib.relu), Matrix{Float32}, Vector{Float32}}, Flux.Dense{typeof(NNlib.relu), Matrix{Float32}, Vector{Float32}}, Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(NNlib.softmax)}}

		should_train::Bool
	end
	
	dict_models = Dict{String, model_struct}(
		"model_NoAug" 		=> model_struct(aug_fun, 								model_NoAug, 	true),
		"model_Rotation" 	=> model_struct(Augmentation.apply_augmentation_rotate, model_Rotation, true),
		"model_Noise" 		=> model_struct(Augmentation.apply_augmentation_noise, 	model_Noise, 	true),
		"model_Flip" 		=> model_struct(Augmentation.apply_augmentation_flip, 	model_Flip, 	true),
		"model_FullAug" 	=> model_struct(Augmentation.apply_augmentation_full, 	model_FullAug, 	true))


	### Check the fully trained model
	if isfile("./models/model_54210.bson")
		@load "./models/model_54210.bson" model_full
	else
		LeNet5.train!(Dict("model_full" => model_struct(aug_fun, model_full, true)), data_full; epochs=epochs_full, batchsize=batchsize_full, lambda=lambda_full, eta=eta_full) # Train the full model
		@save "./models/model_54210.bson" model_full
	end

	
	### Add the fully trained model to the Dict
	dict_models["model_full"] = model_struct(aug_fun, model_full, false)

	
	### Train the models
	LeNet5.train!(dict_models, data_part; epochs=epochs_frac, batchsize=batchsize_frac, lambda=lambda_frac, eta=eta_frac)

	
	__training_finished = rand() # marker that training finished
	nothing
end

# ╔═╡ 37d35f65-6ed8-4eef-b254-5c6d06f01c06
md"## 5 - Evaluation"

# ╔═╡ 49635256-b8b8-4a06-b15e-b77b2c06104e
HTML("<p style='font-size:18px;'>
For evaluation, we measure the model's accuracy (total and per digit) on the test dataset and compare performance across different augmentation methods. This allows us to assess which methods most effectively compensate for the reduced training data and contribute to model accuracy.
</br></br>
After testing with 0.1%, 1%, 10%, 25%, and 50% of the training set, we chose to focus on the 10% subset. It provides a good balance between efficiency and performance, achieving more stable and reliable results (~95% accuracy) compared to 1% (around 60–70%). At the same time, it remains distinguishable from the fully trained model, which reaches about 99% accuracy.
</br></br>
Prediction Table: Shows the total accuracy, accuracy per digit and maximum difference of the 'accuracy per digit' </br>
Confusion Matrix: Shows the counts of true and predicted classifications for each digit, highlighting where the model gets predictions right or wrong </br>
</p>")

# ╔═╡ 9ebf65a7-41f0-48e0-a67e-4789858fdc5e
begin	
	__training_finished # activate after training finished
	
	### Initilize the values for testing
	testingData = LeNet5.getData_test()
	ycold = Flux.onecold(testingData[2], 0:9)


	### Save the testing data
	mutable struct prediction_struct
		preds::Vector{Int64}
		acc_total::Float64
		acc_per_number::Dict{Int64, Tuple{Float64, Int64}}
		max_diff::Float64
	end
	
	dict_model_predictions = Dict{String, prediction_struct}()


	### Test all models
	for (name, model_s) in dict_models
		pred = LeNet5.test(model_s.model, testingData)
		acc = LeNet5.overall_accuracy(pred, ycold)
		accN = LeNet5.accuracy_per_class(pred, ycold)
		v = [val[1] for (key, val) in accN]
		difference = maximum(v)- minimum(v)

		dict_model_predictions[name] = prediction_struct(pred, acc, accN, difference)
	end
	
	testing_finished = rand()
	nothing
end

# ╔═╡ 862df3c3-5275-4a81-abfc-e53238dbe09d
md"### 5.1 - Tables"

# ╔═╡ 5d29528d-8e1c-4d4b-8044-d5b53029251d
md"#### 5.1.1 - Predictions"

# ╔═╡ a8f0b819-0089-48a3-99f3-69797bac2257
md"""
|                   | 0s                                                                                 | 1s                                                                                 | 2s                                                                                 | 3s                                                                                 | 4s                                                                                 | 5s                                                                                 | 6s                                                                                 | 7s                                                                                 | 8s                                                                                 | 9s                                                                                 | Total                                                               | Δ Max                          |
|:-----------------:|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------|--------------------------------|
| Fully Trained     | $(round(dict_model_predictions["model_full"].acc_per_number[0][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[1][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[2][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[3][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[4][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[5][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[6][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[7][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[8][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_per_number[9][1], digits=2))%     | $(round(dict_model_predictions["model_full"].acc_total, digits=2))% | $(round(dict_model_predictions["model_full"].max_diff, digits=2)) |
| No Augmentation   | $(round(dict_model_predictions["model_NoAug"].acc_per_number[0][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[1][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[2][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[3][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[4][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[5][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[6][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[7][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[8][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_per_number[9][1], digits=2))%    | $(round(dict_model_predictions["model_NoAug"].acc_total, digits=2))%  | $(round(dict_model_predictions["model_NoAug"].max_diff, digits=2))  |
| Full Augmentation | $(round(dict_model_predictions["model_FullAug"].acc_per_number[0][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[1][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[2][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[3][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[4][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[5][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[6][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[7][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[8][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_per_number[9][1], digits=2))%  | $(round(dict_model_predictions["model_FullAug"].acc_total, digits=2))% | $(round(dict_model_predictions["model_FullAug"].max_diff, digits=2)) |
| Only Rotation     | $(round(dict_model_predictions["model_Rotation"].acc_per_number[0][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[1][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[2][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[3][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[4][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[5][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[6][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[7][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[8][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_per_number[9][1], digits=2))% | $(round(dict_model_predictions["model_Rotation"].acc_total, digits=2))% | $(round(dict_model_predictions["model_Rotation"].max_diff, digits=2)) |
| Only Noise        | $(round(dict_model_predictions["model_Noise"].acc_per_number[0][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[1][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[2][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[3][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[4][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[5][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[6][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[7][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[8][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_per_number[9][1], digits=2))%    | $(round(dict_model_predictions["model_Noise"].acc_total, digits=2))% | $(round(dict_model_predictions["model_Noise"].max_diff, digits=2))  |
| Only Flip         | $(round(dict_model_predictions["model_Flip"].acc_per_number[0][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[1][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[2][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[3][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[4][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[5][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[6][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[7][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[8][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_per_number[9][1], digits=2))%     | $(round(dict_model_predictions["model_Flip"].acc_total, digits=2))% | $(round(dict_model_predictions["model_Flip"].max_diff, digits=2))  |
"""

# ╔═╡ 5c3dd9a4-603f-435b-b1c4-ba552c262dea
md"#### 5.1.2 - Training Times"

# ╔═╡ f8bf4e08-4857-4d62-9b0e-b5f29faad813
md"""
|                   | Time per Epoch | Total Time |
|-------------------|----------------|------------|
| Fully Trained     | ~4.5s          | ~90s       |
| Fractional (each) | ~0.45s         | ~18s       |

                      														  Trained on M1 Pro, 16GB Ram
"""

# ╔═╡ b85c7f84-d532-44a0-a83f-7ff83d8b3939
md"### 5.2 - Fully Trained Model"

# ╔═╡ 2286f5d2-cf26-415d-a937-31d335734e00
makeFigurePluto_ConfusionMatrix(dict_model_predictions["model_full"].preds, ycold; x_size=600, y_size=600)

# ╔═╡ bcb962ba-15f7-44dd-9deb-d53ed4cc2d9b
md"### 5.3 - Non-Augmented Model"

# ╔═╡ fa6747bf-f695-4635-b00b-0297ef662883
makeFigurePluto_ConfusionMatrix(dict_model_predictions["model_NoAug"].preds, ycold; x_size=600, y_size=600)

# ╔═╡ b87f90ac-3c14-48e2-b83b-9fff926d8119
md"### 5.4 - Fully-Augmented Model"

# ╔═╡ 091e2630-94df-4f6a-922b-2599e845de14
makeFigurePluto_ConfusionMatrix(dict_model_predictions["model_FullAug"].preds, ycold; x_size=600, y_size=600)

# ╔═╡ a5651e12-67d1-484f-9f23-6ca16f93ef00
md"### 5.5 - Only Rotation Model"

# ╔═╡ 80dcc6b6-bf4b-45b9-b137-5c54a1266bb6
makeFigurePluto_ConfusionMatrix(dict_model_predictions["model_Rotation"].preds, ycold; x_size=600, y_size=600)

# ╔═╡ 9391e495-c5a3-4e3c-9210-ca535261b70c
md"### 5.6 - Only Noise Model"

# ╔═╡ 5920c466-ebf5-4d66-9347-10a1ebe29efc
makeFigurePluto_ConfusionMatrix(dict_model_predictions["model_Noise"].preds, ycold; x_size=600, y_size=600)

# ╔═╡ 5ca75737-143e-4a4b-9f48-84ad8f96a832
md"### 5.7 - Only Flip (Mirror) Model"

# ╔═╡ 5bf133d2-b92c-41e9-9ac4-04b2f3a6d873
makeFigurePluto_ConfusionMatrix(dict_model_predictions["model_Flip"].preds, ycold; x_size=600, y_size=600)

# ╔═╡ ee91b0ab-2902-4d86-af4b-d817526daa50
md"## 6 - Conclusion"

# ╔═╡ d782dde0-5d38-423b-82e6-d7505b4dfc58
HTML("<p style='font-size:18px;'>
We found that augmenting the dataset with rotation and noise, individually and in conjunction, helps stabilize the results and reduce variance between runs on average.
Still, with only 10% of the training data, the model's performance is more sensitive to randomness. This leads to greater variability in accuracy across runs. On some runs the non augmented model achieved higher accuracy and less variance than the augmented models. 
This highlights the importance of consistent evaluation and multiple repetitions when working with limited data.
</br></br>
These augmentations should improve the model’s ability to generalize from a smaller dataset, making performance more consistent and closer to that of models trained on the full dataset. </br>
We only tested on the MNIST dataset, which limits our ability to assess generalization, as it does not reflect the diversity and complexity of real-world digit recognition tasks. Future work could invole testing on more challenging data sets with custom handwritten digits, to better evaluate model robustness.
</br></br>
The results of our models could improve if we limit specific augmentation methods to specific digits (e.g. to only use '0' and '8' in our flip (mirror) method), because these digits remain visually similar when mirrored, whereas others (like '2' or '5') can resemble different digits, potentially confusing the model. </br>
Additionally, even more augmentation methods, training for multiple runs to test the stability of the accuracies, and futher improving the models and training algorithms may improve these results.
</p>")

# ╔═╡ dec4ff72-123b-41c7-8136-8f649f4dcaa5
md"## 7 - Further Information"

# ╔═╡ de795b51-bcc5-462d-a78a-f50cd7fb04a5
md"### 7.1 - Data Handling"

# ╔═╡ e940332d-5ed0-4d7a-a0a3-0120e5bae90c
md"""
MNIST dataset: A collection of 28×28 grayscale images of handwritten digits (0–9). Using the Julia package MLDatasets.jl, the data is preprocessed to match the input format expected by the LeNet architecture.

Each image is stored as a tensor of shape (28, 28, 1) — the third dimension representing the grayscale feature map. The dataset is also batched, so the full training set has the shape (28, 28, 1, N), where N is the number of images.

The labels are represented using one-hot encoding: a binary vector of length 10, where the position of the 1 indicates the corresponding digit class (e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] represents the digit 3). This encoding is a standard format used for classification tasks in machine learning.
"""

# ╔═╡ d1b10db1-ffb1-40e0-94a7-95e2a925d921
md"### 7.2 - LeNet5"

# ╔═╡ 7564c7b3-3886-4c18-9597-5e73600c15a5
md"""
LeNet was originally designed to process 32×32 grayscale images; we adapt it here to handle MNIST’s 28×28 images.

Our implementation consists of two convolutional-pooling blocks, followed by three fully connected layers. Each hidden layer uses the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity and helps the network learn complex patterns by zeroing out negative values. The final output layer uses the softmax function to convert the network’s raw outputs into a probability distribution over the 10 digit classes.

$f: \mathbb{R}^{28 \times 28 \times 1} \rightarrow \mathbb{R}^{10} \text{, with the whole function being: }$
$\mathbb{R}^{28 \times 28 \times 1}
\xrightarrow{\text{Conv 1}} \mathbb{R}^{24 \times 24 \times 6}
\xrightarrow{\text{Pool 1 }} \mathbb{R}^{12 \times 12 \times 6}
\xrightarrow{\text{Conv 2 }} \mathbb{R}^{8 \times 8 \times 16}
\xrightarrow{\text{Pool 2}} \mathbb{R}^{4 \times 4 \times 16}
\xrightarrow{\text{Dense}} \mathbb{R}^{256}
\xrightarrow{\text{Dense}} \mathbb{R}^{120}
\xrightarrow{\text{Dense}} \mathbb{R}^{84}
\xrightarrow{\text{Softmax}} \mathbb{R}^{10}$

where the input matrix $\mathbb{R}^{28 \times 28 \times 1}$ is the MNIST image (28 × 28), and the output vector $\mathbb{R}^{10}$ represents the class probabilities for digits 0 through 9 calculated from the last softmax step.
"""

# ╔═╡ fff4a8c8-f44d-4f77-a5bc-e61361ec7cba
md"### 7.3 - Training"

# ╔═╡ fd4cb5e2-af4f-432d-ab04-bb25d3e8efae
md"""
Cross-entropy loss function: Commonly used for multi-class classification problems. Let $\hat{\mathbf{y}}_{i}$ denote the predicted probability distribution for the i-th sample, and $\mathbf{y}_{i}$ the corresponding one-hot encoded ground truth label. The cross-entropy loss over a batch $\mathbb{N}$ of samples is defined as:

$-\frac{1}{N} \sum_{i=1}^{N} y_i \cdot \log(\hat{y}_i)$


To minimize this loss, we use the ADAMW optimizer, a variant of the ADAM algorithm that incorporates weight decay to regularize the model and reduce overfitting. In our implementation, we also apply a weight decay lambda (L2 regularization parameter) to encourage smaller weights, which can improve generalization.
"""

# ╔═╡ 4b121f12-f8b9-47ef-b131-c7be2a4b194e


# ╔═╡ 1e4ae0e7-5c65-4cd7-9f53-0708a8e40351
#=
begin
	fig6 = Figure(size=(900,300))
	ax61 = CairoMakie.Axis(fig6[1,1]; title="Original")
	ax62 = CairoMakie.Axis(fig6[1,2]; title="Noise")
	ax63 = CairoMakie.Axis(fig6[1,3]; title="Augmented")
	hidedecorations!.([ax61, ax62, ax63])
	_noise = 0.1*randn(Float32,(28,28))
	heatmap!(ax61, reshape(xtrain, 28,28,1,:)[:,end:-1:1,1,1], colormap=:grays)
	heatmap!(ax62, _noise[:,end:-1:1,1,1], colormap=:grays)
	heatmap!(ax63, (clamp.(reshape(xtrain, 28,28,1,:)[:,:,1,1] + _noise, 0.0f0, 1.0f0))[:,end:-1:1], colormap=:grays)
	fig6
end
=#

# ╔═╡ f90ab7e0-1554-463b-8545-870f6f6d469d
#=
begin
	fig5 = Figure(size = (450, 450))
	
	# Create an axis for the heatmap
	ax5 = Axis(fig5[1, 1], 
	          title = "Label = 1, Prediction = $(onecold(model(im0), 0:9)[1])", )
	hidedecorations!(ax5)
	heatmap!(ax5,im0[:,end:-1:1,1,1], colormap = :grays, colorrange = (0, 1))
	fig5
end
=#

# ╔═╡ 65652a6f-8299-4e0d-991d-c2c4abf35821
html"""
<style>
@media screen {
	main {
		margin: 0 auto;
		max-width: 100vw;
		padding-left: 2%;
		padding-right: 350px;
	}
}
</style>
"""

# ╔═╡ Cell order:
# ╠═ba491ad9-0c09-44a3-ab98-21919da7c62e
# ╟─d1cbf365-b3b1-494f-8158-f7c3b33a9fd3
# ╟─a8cfdd5a-c5ef-4ef0-946c-d9594724acc3
# ╟─9dc2706a-7b71-4576-bb04-68c7aacbb9ad
# ╟─dc6459b8-92a2-45c1-8378-5089def92f15
# ╟─4b142462-d76d-4948-8de0-b32b65f4f0b7
# ╟─51c66468-da97-42a0-b594-e1a531ce66a2
# ╟─132df693-22dc-4757-8f21-5421317f836b
# ╟─f5762b3f-37af-45f0-aecc-d7785b58984b
# ╟─bd8f9222-1c39-45a0-b29f-c9e2b7018e79
# ╟─4eab70fe-25be-44a3-956c-899e5712a790
# ╟─9ed2b7e1-9da8-49d6-a463-b23d452d1ee7
# ╟─d112f3f9-700a-4d40-882a-db90c3ed7706
# ╟─b5420a56-02c0-4f9a-9a80-457d039dd1f3
# ╟─1b55e14d-465d-4458-9d68-5d4da206a226
# ╟─085188be-e85f-476c-9aff-fe87a4dcf423
# ╟─ae063e85-0ed0-40b1-b84c-82e3e6de0d1f
# ╟─34189beb-75e1-4bec-acf8-d726515e80e6
# ╟─3152263a-e1b5-43c8-b957-9f50c66b2fc4
# ╟─efc1f254-e5b0-4cc6-94f3-13ca9c77aa9c
# ╠═22cddec6-24bc-4afd-9efb-a428880355ea
# ╟─146b2db4-3834-453d-887e-e70cc7f9de20
# ╟─33da904c-a964-4305-a5cd-eff2fe6543c9
# ╟─37d35f65-6ed8-4eef-b254-5c6d06f01c06
# ╟─49635256-b8b8-4a06-b15e-b77b2c06104e
# ╟─9ebf65a7-41f0-48e0-a67e-4789858fdc5e
# ╟─862df3c3-5275-4a81-abfc-e53238dbe09d
# ╟─5d29528d-8e1c-4d4b-8044-d5b53029251d
# ╟─a8f0b819-0089-48a3-99f3-69797bac2257
# ╟─5c3dd9a4-603f-435b-b1c4-ba552c262dea
# ╟─f8bf4e08-4857-4d62-9b0e-b5f29faad813
# ╟─b85c7f84-d532-44a0-a83f-7ff83d8b3939
# ╟─2286f5d2-cf26-415d-a937-31d335734e00
# ╟─bcb962ba-15f7-44dd-9deb-d53ed4cc2d9b
# ╟─fa6747bf-f695-4635-b00b-0297ef662883
# ╟─b87f90ac-3c14-48e2-b83b-9fff926d8119
# ╟─091e2630-94df-4f6a-922b-2599e845de14
# ╟─a5651e12-67d1-484f-9f23-6ca16f93ef00
# ╟─80dcc6b6-bf4b-45b9-b137-5c54a1266bb6
# ╟─9391e495-c5a3-4e3c-9210-ca535261b70c
# ╟─5920c466-ebf5-4d66-9347-10a1ebe29efc
# ╟─5ca75737-143e-4a4b-9f48-84ad8f96a832
# ╟─5bf133d2-b92c-41e9-9ac4-04b2f3a6d873
# ╟─ee91b0ab-2902-4d86-af4b-d817526daa50
# ╟─d782dde0-5d38-423b-82e6-d7505b4dfc58
# ╟─dec4ff72-123b-41c7-8136-8f649f4dcaa5
# ╟─de795b51-bcc5-462d-a78a-f50cd7fb04a5
# ╟─e940332d-5ed0-4d7a-a0a3-0120e5bae90c
# ╟─d1b10db1-ffb1-40e0-94a7-95e2a925d921
# ╟─7564c7b3-3886-4c18-9597-5e73600c15a5
# ╟─fff4a8c8-f44d-4f77-a5bc-e61361ec7cba
# ╟─fd4cb5e2-af4f-432d-ab04-bb25d3e8efae
# ╟─4b121f12-f8b9-47ef-b131-c7be2a4b194e
# ╟─1e4ae0e7-5c65-4cd7-9f53-0708a8e40351
# ╟─f90ab7e0-1554-463b-8545-870f6f6d469d
# ╟─65652a6f-8299-4e0d-991d-c2c4abf35821
