### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

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

# ╔═╡ 8cb672c0-5358-11f0-16ec-75bbd3266681
md"""
# Handwritten Digit Recognition using the MNIST dataset

Handwritten digit recognition is a well-known task in the field of machine learning. In this project, we use the MNIST dataset, which contains thousands of images of handwritten digits (0 through 9), to train a model that can automatically recognize and classify these digits.

Our project is a great starting point for exploring machine learning as it walks through the full pipeline — from loading and preprocessing data, to model training, evaluation, and interpretation. Along the way, you’ll gain hands-on experience with typical workflows and techniques used in real-world ML projects.

Handwritten digit recognition is a well-known task in the field of machine learning. In this project, we use the MNIST dataset to train a simple convolutional neural network (LeNet) that can classify digits from 0 to 9. Rather than focusing on model architecture, the goal is to explore how the amount and quality of training data affect model performance. We experiment with different training set sizes and apply data augmentation techniques to improve accuracy when using limited data. This provides practical insight into how data influences learning outcomes and highlights the importance of preprocessing and augmentation in real-world Machine Learning workflows.
"""

# ╔═╡ 8c7603fa-07a7-4435-b80b-562f7ada38ee
md"""
## 1. Input Data Handling for LeNet Using The MNIST Dataset

We start with loading the training and test data from the MNIST dataset, a collection of 28×28 grayscale images of handwritten digits (0–9). Using the Julia package MLDatasets.jl, the data is preprocessed to match the input format expected by the LeNet architecture.


"""

# ╔═╡ 4bd85dd1-6661-4c23-a1c9-389da49187e8
begin
	data_part = LeNet5.getData_train(; amounts=fill(542,10))
	data_full = LeNet5.getData_train(; amounts=fill(5421,10))

	println(size(data_part[2]))

	data_finished = rand() # marker that the data sets are prepared
end

# ╔═╡ a1e56527-4065-477a-941d-9afa5d8c5628
md"# TODO: HEADER AND DESC FOR VISUALIZE"

# ╔═╡ 384bd25d-66f6-482e-935d-9c9708179690
@time begin
	# get a small (10 each) data set to visualize the augmentation
	data_small = LeNet5.getData_train(; amounts=fill(1,10))

	# apply all kinds of augmentation
	(data_rotate_x, data_rotate_y)= Augmentation.apply_augmentation_rotate(data_small[1], data_small[2])
	(data_noise_x, data_noise_y)= Augmentation.apply_augmentation_noise(data_small[1], data_small[2])
	(data_flip_x, data_flip_y)= Augmentation.apply_augmentation_flip(data_small[1], data_small[2])
	(data_all_x, data_all_y)= Augmentation.apply_augmentation_full(data_small[1], data_small[2])

	# create all figures
	fig_aug_rot = LeNet5.makeFigurePluto_Images(1200,150,data_rotate_x, data_rotate_y)
	fig_aug_noise = LeNet5.makeFigurePluto_Images(1200,150,data_noise_x, data_noise_y)
	fig_aug_flip = LeNet5.makeFigurePluto_Images(1200,150,data_flip_x, data_flip_y)
	fig_aug_full = LeNet5.makeFigurePluto_Images(1200,150,data_all_x, data_all_y)
	fig_no_aug = LeNet5.makeFigurePluto_Images(1200,150,data_small[1],data_small[2])
end

# ╔═╡ 048d5fb2-8cf8-4a61-bfc8-c97fedfa30d8
fig_aug_rot

# ╔═╡ dc548758-a391-4fce-8c9c-5b4ccfd04994
fig_aug_noise

# ╔═╡ 04a45c1c-79a8-4d72-8cae-aeedef777f69
fig_aug_flip

# ╔═╡ ee180bc9-afd1-4fb5-8d3c-e74eec4829a1
fig_aug_full

# ╔═╡ 64fdd548-0994-4ccb-8a91-dc394f478926
md"""
Each image is stored as a tensor of shape (28, 28, 1) — the third dimension representing the grayscale feature map. The dataset is also batched, so the full training set has the shape (28, 28, 1, N), where N is the number of images.

The labels are represented using one-hot encoding: a binary vector of length 10, where the position of the 1 indicates the corresponding digit class (e.g., [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] represents the digit 3). This encoding is a standard format used for classification tasks in machine learning.
"""

# ╔═╡ 94da4f46-f348-4148-b7c5-7e01b901f356
md"""
## 2. LeNet-Based Convolutional Neural Network
"""

# ╔═╡ 69ae3d42-9d1b-4393-9b88-9a3930b03ac9
md"""
To classify handwritten digits from the MNIST dataset, we implement a convolutional neural network based on the LeNet-5 architecture. LeNet was originally designed to process 32×32 grayscale images; we adapt it here to handle MNIST’s 28×28 images.

Our implementation consists of two convolutional-pooling blocks, followed by three fully connected layers. Each hidden layer uses the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity and helps the network learn complex patterns by zeroing out negative values. The final output layer uses the softmax function to convert the network’s raw outputs into a probability distribution over the 10 digit classes.

$f: \mathbb{R}^{784}\rightarrow \mathbb{R}^{10}$
where the input vector $x \in \mathbb{R}^{784}$ is the flattened image (28 × 28), and the output vector $y \in \mathbb{R}^{10}$ represents the class probabilities for digits 0 through 9.

The model is constructed using the Flux.Chain function, which sequentially applies the listed layers to the input. The Dense layers represent fully connected layers, while Conv and MaxPool handle the convolutional and downsampling operations.
"""

# ╔═╡ 4016bf0e-e127-46ed-bd57-74d6e46866b3
md"""
$\mathbb{R}^{28 \times 28 \times 1}
\xrightarrow{\text{Conv 1}} \mathbb{R}^{24 \times 24 \times 6}
\xrightarrow{\text{Pool 1 }} \mathbb{R}^{12 \times 12 \times 6}
\xrightarrow{\text{Conv 2 }} \mathbb{R}^{8 \times 8 \times 16}
\xrightarrow{\text{Pool 2}} \mathbb{R}^{4 \times 4 \times 16}$
"""

# ╔═╡ d28a4019-56f3-451b-8d86-676e410c9113
md"""
$\xrightarrow{\text{Flatten}} \mathbb{R}^{256}
\xrightarrow{\text{Dense}} \mathbb{R}^{120}
\xrightarrow{} \mathbb{R}^{84}
\xrightarrow{} \mathbb{R}^{10}
\xrightarrow{\text{softmax}} \text{probabilities}$
"""

# ╔═╡ 299f8e36-41ab-4536-91d8-5d4cac16aecb
md"""
Each transformation is the result of learned filters or weight matrices applied to the data, followed by nonlinear activations or downsampling. The softmax layer at the end converts the final activations into class probabilities.
"""

# ╔═╡ 06baaad7-d8b2-4a51-8421-2f8d1a8ae70b
md"""
## 3. Training the Network Using Cross-Entropy Loss and the ADAM Optimizer
With the model architecture in place, we now train the convolutional neural network on the MNIST dataset. The training process involves adjusting the model’s internal parameters — weights and biases — to minimize the prediction error on the training data. This is achieved by defining a loss function and applying an optimization algorithm that updates the parameters to reduce this loss.

For this task, we employ the cross-entropy loss function, which is commonly used for multi-class classification problems. Let $\hat{\mathbf{y}}_{i}$ denote the predicted probability distribution for the i-th sample, and $\mathbf{y}_{i}$ the corresponding one-hot encoded ground truth label. The cross-entropy loss over a batch $\mathbb{N}$ of samples is defined as:
"""

# ╔═╡ b5cb9b39-a573-4303-b4b4-760b029ac99e
begin #= only for combined augmentation
	(data_rotate_x, data_rotate_y)= Augmentation.apply_augmentation_rotate(data_part[1], data_part[2])
	(data_noise_x, data_noise_y)= Augmentation.apply_augmentation_noise(data_part[1], data_part[2])
	(data_all_x, data_all_y)= Augmentation.apply_augmentation_full(data_part[1], data_part[2])

	x_dim = ndims(data_part[1])
	y_dim = ndims(data_part[2])
	
	cdata_rotate_x = cat(data_part[1], data_rotate_x; dims=x_dim)
	cdata_rotate_y = cat(data_part[2], data_rotate_y; dims=y_dim)
	
	cdata_noise_x = cat(data_part[1], data_noise_x; dims=x_dim)
	cdata_noise_y = cat(data_part[2], data_noise_y; dims=y_dim)
	
	cdata_all_x = cat(data_part[1], data_all_x; dims=x_dim)
	cdata_all_y = cat(data_part[2], data_all_y; dims=y_dim)
	=#
end

# ╔═╡ 0377fa30-04f9-452e-8d3a-c32f9635a7ad
begin
	data_finished # start after the data sets are prepared 
		
	model_NoAug = LeNet5.createModel() 
	model_Rotation = LeNet5.createModel() 
	model_Noise = LeNet5.createModel() 
	model_Flip = LeNet5.createModel()
	model_FullAug = LeNet5.createModel() 
	model_full = LeNet5.createModel()

	if isfile("./models/model_54210.bson")
		@load "./models/model_54210.bson" model_full
	else
		LeNet5.train!(model_full, data_full)
		@save "./models/model_54210.bson" model_full
	end

	aug_fun::Function = (a, b) -> (a,b)
	
	dict_models_funs = Dict(model_NoAug => aug_fun,
							model_Rotation => Augmentation.apply_augmentation_rotate,
							model_Noise => Augmentation.apply_augmentation_noise,
							model_Flip => Augmentation.apply_augmentation_flip,
							model_FullAug => Augmentation.apply_augmentation_full
						   )

	#=
	LeNet5.train!(model_NoAug, data_part;batchsize=4, epochs=40, lambda=1e-2, eta=3e-4)
	LeNet5.train!(model_Rotation, data_part;batchsize=4, epochs=40, aug_fun=Augmentation.apply_augmentation_rotate, chance=1, lambda=1e-2, eta=3e-4)
	LeNet5.train!(model_Noise, data_part;batchsize=32, epochs=40, aug_fun=Augmentation.apply_augmentation_noise, chance=1, lambda=1e-2, eta=3e-4)
	LeNet5.train!(model_FullAug, data_part;batchsize=32, epochs=40, aug_fun=Augmentation.apply_augmentation_full, chance=1, lambda=1e-2, eta=3e-4)
	=#
	LeNet5.train!(dict_models_funs, data_part; batchsize=32, epochs=30, lambda=1e-2, eta=3e-4)

	training_finished=rand() # marker that training finished
end

# ╔═╡ fe56b42b-28db-4dfc-84d6-1d4e80c6aaa0
md"""
$-\frac{1}{N} \sum_{i=1}^{N} y_i \cdot \log(\hat{y}_i)$
"""

# ╔═╡ 83d17c51-34d2-4cc1-95b0-0a405c9f1499
md"""
This loss measures the dissimilarity between the predicted and true distributions, penalizing incorrect predictions more heavily when the confidence is high.

To minimize this loss, we use the ADAMW optimizer, a variant of the ADAM algorithm that incorporates weight decay to regularize the model and reduce overfitting. In our implementation, we also apply WeightDecay(settings_lambda) to encourage smaller weights, which can improve generalization.

The training loop proceeds over a specified number of epochs. In each epoch, the training data is divided into mini-batches, and for each batch, the following steps are performed:

1. The model computes predictions for the current batch.
2. The loss is computed using the cross-entropy function.
3. Gradients are calculated via backpropagation.
4. The optimizer updates the model parameters using the computed gradients.

The loss value at each step is recorded to monitor training progress.
"""

# ╔═╡ 3f93e564-f110-40db-85ee-58bfd31c791e
md"""
The following block initializes the model and runs the training procedure for 20 epochs with a batch size of 32:
"""

# ╔═╡ 64c2570c-b055-4c8f-bb97-df17e0adaef4
#=
begin
	fig4 = Figure(size=(450,450))
	ax = CairoMakie.Axis(
		fig4[1,1]; 
		xlabel="training steps", 
		ylabel="loss",
		title="Training history",
		# yscale=log10,
	)
	
	p = lines!(ax, loss_history)
	fig4
end
=#

# ╔═╡ dc63e616-488b-45de-b8c3-5449c1a58267
md"""
This training setup illustrates a typical supervised learning pipeline using modern deep learning tools. It highlights how the combination of an appropriate loss function, optimizer, and regularization strategy can lead to effective model learning on image classification tasks.
"""

# ╔═╡ 917fa567-43b4-4a5a-a7dd-52de16e4651f
md"""
## 4. Model Evaluation and Prediction on the Test Set
To evaluate the trained model, we apply it to the MNIST test set and generate predictions. The model outputs a probability distribution over the 10 digit classes for each image, and we select the most likely class using onecold:
"""

# ╔═╡ 9ebf65a7-41f0-48e0-a67e-4789858fdc5e
begin	
	training_finished # activate after training finished
	
	testingData = LeNet5.getData_test()
	ycold = Flux.onecold(testingData[2], 0:9)

	# Training the models differently

	models_dict = Dict("model_NoAug" => model_NoAug,
					   "model_Rotation" => model_Rotation,
					   "model_Noise" => model_Noise,
					   "model_Flip" => model_Flip,
					   "model_FullAug" => model_FullAug,
					   "model_Full" => model_full
					   )


	for (name,model) in models_dict
		
		pred = LeNet5.test(model, testingData)
		acc = LeNet5.overall_accuracy(pred,ycold)
		println("$(name) T_Acc: $(acc)")
		accN = LeNet5.accuracy_per_class(pred, ycold)
		
		v = Float32[]
		for (key, val) in accN
			println("     Num: $(key) -> Acc: $(round(val[1], digits=2)) for $(val[2])")			
			push!(v, val[1])
		end
		
		println("     $(round(maximum(v)- minimum(v), digits=2))")
		
	end
	testing_finished= rand()
end

# ╔═╡ 233b873b-d5bb-49cd-a432-6be2be754387
# @bind plotslice2 PlutoUI.Slider(1:div(size(ytest,2),12))

# ╔═╡ 3e4013a4-dc76-4715-8e51-12cd24c48949
md"""
We visualize a selection of 12 test images along with their predicted labels. Each image is displayed in grayscale, and the predicted digit is shown in the title:
"""

# ╔═╡ 3aea78df-5212-4ae8-8c67-6406d8c20591
#=
begin
	indices2 = 12 * (plotslice2 - 1) + 1 : 12 * plotslice2
	fig2 = Figure(size = (800, 600), fontsize=20)
	for (i, idx) in enumerate(indices2)
	    ax2 = Axis(fig2[(i-1)÷4+1, (i-1)%4+1], title="pred=$(preds[idx])")
		hidedecorations!(ax2)
	    heatmap!(ax2, reshape(xtest,28,28,1,:)[:,end:-1:1,1,idx], colormap = :grays, colorrange = (0, 1))
	end
	fig2
end
=#

# ╔═╡ 30d0da08-634b-4334-ac8a-fc1eefeac7da
md"""
This visual inspection offers an intuitive way to assess prediction quality and spot potential misclassifications.
"""

# ╔═╡ f63eaf02-ee42-467c-b00f-623582c5dac0
md"""
## 5. Conclusion

This project demonstrates how a simple convolutional neural network can be trained for handwritten digit recognition using Julia and the Flux library. While the implementation serves as a minimal working example, it omits several important aspects of practical machine learning workflows, including:

* **Model evaluation:** How to quantitatively monitor performance, measure accuracy, and determine when to stop training.
* **Hyperparameter tuning:** Adjusting parameters such as learning rate and regularization strength to optimize results.
* **Robustness and generalization:** Analyzing class-wise accuracy and testing how the model performs on data that differs from the training set.
* **Architectural improvements:** Exploring deeper or more advanced network structures to enhance performance.

These topics provide natural next steps for improving both model quality and experimental rigor in more advanced projects.
"""

# ╔═╡ d183f9cc-a22f-487b-99ae-e8b60166e4b6
md"""
## 6. Outlook 
### Model Evaluation Using a Confusion Matrix
To better understand the model’s performance, we construct a confusion matrix. This matrix compares predicted labels to the true labels, showing how many instances of each digit class were correctly or incorrectly classified.

The function below computes the confusion matrix by counting how often a true class j was predicted as class i:
"""

# ╔═╡ db1f1008-27d5-4d35-ac06-05e61f86f692
md"""
We then visualize the matrix as a heatmap, with annotations to indicate the number of samples in each cell:
"""

# ╔═╡ 350cc27f-995b-4064-9bfa-de0a3ea05ac1
md"""
From the confusion matrix, we observe that the model correctly classifies most digits without any strong systematic bias. However, the overall accuracy, defined as the ratio of correct predictions to the total number of predictions, is:
"""

# ╔═╡ 21bc1b1b-3319-443b-9401-4a4c5cba6e4f
begin
	pred_rot = LeNet5.test(model_Rotation, testingData)
	makeFigurePluto_ConfusionMatrix(pred_rot, ycold; x_size=600, y_size=600)
end

# ╔═╡ 5920c466-ebf5-4d66-9347-10a1ebe29efc
begin
	pred_noise = LeNet5.test(model_Noise, testingData)
	makeFigurePluto_ConfusionMatrix(pred_noise, ycold; x_size=600, y_size=600)
end

# ╔═╡ 5bf133d2-b92c-41e9-9ac4-04b2f3a6d873
begin
	pred_flip = LeNet5.test(model_Flip, testingData)
	makeFigurePluto_ConfusionMatrix(pred_flip, ycold; x_size=600, y_size=600)
end

# ╔═╡ 091e2630-94df-4f6a-922b-2599e845de14
begin
	pred_aug_full = LeNet5.test(model_FullAug, testingData)
	makeFigurePluto_ConfusionMatrix(pred_aug_full, ycold; x_size=600, y_size=600)
end

# ╔═╡ fa6747bf-f695-4635-b00b-0297ef662883
begin
	pred_no_aug = LeNet5.test(model_NoAug, testingData)
	makeFigurePluto_ConfusionMatrix(pred_no_aug, ycold; x_size=600, y_size=600)
end

# ╔═╡ 2286f5d2-cf26-415d-a937-31d335734e00
begin
	pred = LeNet5.test(model_full, testingData)
	makeFigurePluto_ConfusionMatrix(pred, ycold; x_size=600, y_size=600)
end

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
# ╟─8cb672c0-5358-11f0-16ec-75bbd3266681
# ╠═ba491ad9-0c09-44a3-ab98-21919da7c62e
# ╟─8c7603fa-07a7-4435-b80b-562f7ada38ee
# ╠═4bd85dd1-6661-4c23-a1c9-389da49187e8
# ╟─a1e56527-4065-477a-941d-9afa5d8c5628
# ╟─384bd25d-66f6-482e-935d-9c9708179690
# ╠═048d5fb2-8cf8-4a61-bfc8-c97fedfa30d8
# ╠═dc548758-a391-4fce-8c9c-5b4ccfd04994
# ╠═04a45c1c-79a8-4d72-8cae-aeedef777f69
# ╠═ee180bc9-afd1-4fb5-8d3c-e74eec4829a1
# ╟─64fdd548-0994-4ccb-8a91-dc394f478926
# ╟─94da4f46-f348-4148-b7c5-7e01b901f356
# ╟─69ae3d42-9d1b-4393-9b88-9a3930b03ac9
# ╟─4016bf0e-e127-46ed-bd57-74d6e46866b3
# ╟─d28a4019-56f3-451b-8d86-676e410c9113
# ╟─299f8e36-41ab-4536-91d8-5d4cac16aecb
# ╟─06baaad7-d8b2-4a51-8421-2f8d1a8ae70b
# ╠═b5cb9b39-a573-4303-b4b4-760b029ac99e
# ╠═0377fa30-04f9-452e-8d3a-c32f9635a7ad
# ╟─fe56b42b-28db-4dfc-84d6-1d4e80c6aaa0
# ╟─83d17c51-34d2-4cc1-95b0-0a405c9f1499
# ╟─3f93e564-f110-40db-85ee-58bfd31c791e
# ╟─64c2570c-b055-4c8f-bb97-df17e0adaef4
# ╟─dc63e616-488b-45de-b8c3-5449c1a58267
# ╟─917fa567-43b4-4a5a-a7dd-52de16e4651f
# ╠═9ebf65a7-41f0-48e0-a67e-4789858fdc5e
# ╟─233b873b-d5bb-49cd-a432-6be2be754387
# ╟─3e4013a4-dc76-4715-8e51-12cd24c48949
# ╟─3aea78df-5212-4ae8-8c67-6406d8c20591
# ╟─30d0da08-634b-4334-ac8a-fc1eefeac7da
# ╟─f63eaf02-ee42-467c-b00f-623582c5dac0
# ╟─d183f9cc-a22f-487b-99ae-e8b60166e4b6
# ╟─db1f1008-27d5-4d35-ac06-05e61f86f692
# ╟─350cc27f-995b-4064-9bfa-de0a3ea05ac1
# ╠═21bc1b1b-3319-443b-9401-4a4c5cba6e4f
# ╠═5920c466-ebf5-4d66-9347-10a1ebe29efc
# ╠═5bf133d2-b92c-41e9-9ac4-04b2f3a6d873
# ╠═091e2630-94df-4f6a-922b-2599e845de14
# ╠═fa6747bf-f695-4635-b00b-0297ef662883
# ╠═2286f5d2-cf26-415d-a937-31d335734e00
# ╠═1e4ae0e7-5c65-4cd7-9f53-0708a8e40351
# ╟─f90ab7e0-1554-463b-8545-870f6f6d469d
# ╟─65652a6f-8299-4e0d-991d-c2c4abf35821
