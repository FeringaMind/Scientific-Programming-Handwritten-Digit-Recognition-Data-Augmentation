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

# ╔═╡ 793b0fa9-28f5-4916-9732-6d953d9aa22f
begin
	using Pkg, Flux, CUDA, MLDatasets, PlutoUI, CairoMakie, LinearAlgebra, FileIO
	Pkg.activate("")

	import Flux: DataLoader, onehotbatch, onecold, crossentropy
	
	CairoMakie.activate!(; px_per_unit = 4)
	PlutoUI.TableOfContents()
end

# ╔═╡ 4656940a-c9c7-11eb-186d-8bd3fedf80e2
md"""
# Handwritten Digit Recognition using the MNIST dataset

Handwritten digit recognition is a popular task that can be tackled using machine learning techniques. In this project, we will use the MNIST dataset to train a model for such a task. This can serve as an introduction to the field of machine learning and allows you to get familiar with the workflow and analyses typically performed when working on a machine learning project.
"""

# ╔═╡ 6b6e78ce-4ac8-4caf-8a37-5ea74a124830
md"""
## 1. Loading the MNIST dataset (Input Data Handling for LeNet)

Let's start by loading the train and test data that make up the MNIST dataset.
"""

# ╔═╡ 00b0d98c-375a-4e6c-9a1b-fab2d53bfd18
function getdata(datadir)
	# Disable manual confirmation of dataset download
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
	xtrain, ytrain = MLDatasets.MNIST(Float32, dir=datadir, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Float32, dir=datadir, split=:test)[:]
	
    # Reshape Data in order to flatten each image into a linear array
    xtrain = reshape(xtrain, 28,28,1,:) #|> Flux.flatten #Flux Syntax to flatten
	xtest = reshape(xtest, 28,28,1,:) #|> Flux.flatten

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    return xtrain, ytrain, xtest, ytest
end

# ╔═╡ 5be0a82d-14e7-4a3d-b1ac-cf25a7b70093
xtrain, ytrain, xtest, ytest = getdata("mnist_data")

# ╔═╡ abc054bc-2928-4088-8bfe-346724e6e36a
md"""
As we can see, each feature is a vector of length 784, which corresponds to an image of size $28\times 28$. The labels are vectors of length 10, consisting of 9 0s and one 1. Here, the position of the 1 encodes the class of the given data, i.e., the digit encoded by the image. This is called one-hot encoding.
"""

# ╔═╡ 080667aa-0c5e-4c9a-89f8-90489b114d19
@bind plotslice PlutoUI.Slider(1:div(size(ytest,2),12))

# ╔═╡ cde625f9-f6ca-48d7-ae31-4ec93bda748b
begin
	indices = 12 * (plotslice - 1) + 1 : 12 * plotslice
	fig = Figure(size = (800, 600), fontsize=20)
	for (i, idx) in enumerate(indices)
	    ax = Axis(fig[(i-1)÷4+1, (i-1)%4+1], title = "label=$(onecold(ytest[:, idx], 0:9))")
		hidedecorations!(ax)
	    heatmap!(ax, reshape(xtest, 28,28,1,:)[:,end:-1:1,1,idx], colormap = :grays, colorrange = (0, 1))
	end
	fig
end

# ╔═╡ 89968d90-d65d-43c7-ba89-95da468628c5
md"""
## 2.2 Building a model

Goal for this part: (see (https://en.wikipedia.org/wiki/LeNet))

C5 Layer - Output (Current state: Input -> C5 -> Output)

$f: \mathbb{R}^{784}\rightarrow \mathbb{R}^{10}$



Next, we build a neural network with one hidden layer. This can be seen as a function of $f: \mathbb{R}^{784}\rightarrow \mathbb{R}^{84}\rightarrow \mathbb{R}^{10}$. The input to the function is the vectorized images. The output is a vector of numbers that indicate to which class the image belongs. The class with the largest number will be the class predicted by the network.

For simplicity, we will use a network consisting of a single fully connected layer with a sigmoid activation function. Mathematically, such a network can be written as
$$f(\mathbf{x}) = \sigma(\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2)$$, where $\sigma$ is the softmax function.

The function is parameterized by the weights $\mathbf{W}_1\in\mathbb{W}^{84\times784}$,
$\mathbf{W}_2\in\mathbb{W}^{10\times84}$
and the biases
$\mathbf{b}_1\in\mathbb{R}^{84}$,
$\mathbf{b}_2\in\mathbb{R}^{10}$.
"""

# ╔═╡ 4cad4ec3-08d4-4b57-ba49-4a9a63d89aff
function build_model_NN() 
	
    return Chain(
		# C1 Convolution Layer: 28x28x1 (+2 padding) => 28x28x6
		Conv((5, 5), 1=>6, pad=(2,2), relu),

		# S2 Pooling Layer: 28x28x6 => 14x14x6
		x -> maxpool(x, (2,2)),
	
		# C3 Convolution Layer: 14x14x6 => 10x10x16
		Conv((5, 5), 6=>16, pad=(0,0), relu),

		# S4 Pooling Layer: 10x10x16 => 5x5x16
		x -> maxpool(x, (2,2)),
	
		# C3 Convolution Layer: 5x5x16 => 1x1x120
		Conv((5, 5), 16=>120, pad=(0,0), relu),

		# Reshape to (120, 128)
		x -> reshape(x, :, size(x, 4)),
	
		# F6 Dense layer: 120 => 84
		Dense(120, 84), relu, 	
	
		# Output Dense layer: 84 => 10
		Dense(84, 10),	
	
		# Softmax to create a probability distribution
		softmax
	) 
end

# ╔═╡ 7be6f10f-8a32-4cca-922d-198e5e4efd71
md"
The calculation in a layer is done by the Flux.Dense function, the Flux.Chain function connects the layers in order, and then we use the beforementioned NNlib.softmax function to calculate the desired output.
"

# ╔═╡ 94666008-bfa9-4b50-a84b-aa7d92e1a93f
md"""
## 3. Training the model !Rework for our Purposes (TODO)!

goal: using the RELU, Adam Optimizer

When training the network, we present the network with pairs of features and labels. Based on these, the weights and biases of the network are then updated to reduce the prediction error. To update the weights, we need to specify a loss function. In the training process, an optimizer will change the weights so that the given loss function is minimized.

Here we will use the cross-entropy loss. Let $\hat{\mathbf{y}}$ denote the outputs of our model and $\mathbf{y}$ the corresponding label. Then, our loss function can be written as 

$-\frac{1}{N}\sum_{i=1}^N y_i\cdot log(\hat{y}_i)$.
"""

# ╔═╡ 363f8093-4960-4c67-a444-fba097de2a13
loss = crossentropy

# ╔═╡ 13f609b2-c986-4494-bb08-72f5903c395e
md"""
A simple way to perform optimization is to use stochastic gradient descent.
"""

# ╔═╡ 4b9c3e33-43e4-45f1-b8bc-63c0236e9179
opt = ADAM(1e-3)

# ╔═╡ a24a2469-9081-459e-8382-5c8ea0d4a82e
md"""
Having build our model and specified both loss function and optimizer let us now perform training
"""

# ╔═╡ d1ee8eed-5a36-4050-a733-3455b16bb833
function train!(model, data; epochs=100, batchsize=128, 
				device=gpu, optimizer=Descent(1e-3))
	
	# setup data and model
	train_loader = DataLoader(data, batchsize=batchsize, shuffle=true)
	loss_function = crossentropy
	opt_state = Flux.setup(optimizer, model)
	
	# perform training
	loss_history = []
	for epoch in 1:epochs
		for xy_cpu in train_loader
			x, y = xy_cpu |> device	# transfer data to device
			loss, grads = Flux.withgradient(model) do m
				y_hat = m(x)
				loss_function(y_hat, y)
			end
			push!(loss_history, loss)
			Flux.update!(opt_state, model, grads[1]) # update parameters pf model
		end
	end
	
	return loss_history
end

# ╔═╡ 9eb390a8-6dbb-45de-80b0-fb5f072e66c9
begin
	device = gpu
	model = build_model_NN() |> device
	loss_history = train!(model, (xtrain, ytrain); epochs=20, device)
end

# ╔═╡ 24223983-08b4-4966-a3aa-812b6fa2ccda
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

# ╔═╡ 92d96f31-0292-483b-9581-77306624ef28
md"""
## 4. Evaluating the model

Let us use our model to predict the labels of our test set
"""

# ╔═╡ a8ca0f30-e936-45b2-9aad-8259a1ba9a72
preds = onecold(model(xtest |> device) |> gpu, 0:9)

# ╔═╡ 133db6fb-e7f8-4fa7-ad84-22a4fb5c818b
@bind plotslice2 PlutoUI.Slider(1:div(size(ytest,2),12))

# ╔═╡ efc55b09-af32-430b-9429-ddfdcfde2f3f
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

# ╔═╡ d5fc3650-21da-418a-ba33-422d4b33b625
md"""
## 5. Conclusion
As we have seen, simple models for handwritten digit recognition can be trained quite easily using the Julia library Flux.
Note, however, that this is merely a minimal working example. Many important aspects are not covered here. These include
* How to monitor and test the accuracy of your model? When to stop training?
* How to tune hyperparameters, such as the learning rate of your optimizer? 
* A more detailed evaluation of the model. What is the overall accuracy? Is the accuracy the same for all classes? How robust is the model? What happens when you use it with "different" data?
* How can the model be improved by using more complex network architectures?
"""

# ╔═╡ d9cb5a95-e424-4f61-9aa8-643d5d5ceab0
md"""
## 6. Outlook
### Evaluation

A simple way to analyze the performance of a given classifier is to look at the resulting **confusion matrix**. This matrix allows you to see directly how many instances of each class have been labeled with a particular class.
"""

# ╔═╡ 9b36e662-52de-4c5f-b4ca-00d58399f072
begin
	function confusionMatrix(ŷ, y)
			confMat = zeros(Float64, (10,10))
			for i=0:9, j=0:9
				 confMat[i+1,j+1] = sum((ŷ .== i) .&& (y .== j))
			end
			return confMat
	end
	
	confMat = confusionMatrix(preds, onecold(ytest, 0:9))
end;

# ╔═╡ c7b47cfa-71a3-4df4-b86f-d0699365982e
begin
	# Create a figure
	fig3 = Figure(size = (600, 600))
	
	# Create an axis for the heatmap
	ax3 = Axis(fig3[1, 1], 
	          title = "Confusion Matrix", 
	          xlabel = "predicted digit", 
	          ylabel = "true digit", 
	          xticks = 0:9, 
	          yticks = 0:9)
	
	# Plot the heatmap
	hm = heatmap!(ax3, 0:9, 0:9, confMat, colormap = :viridis)
	
	# Annotate the heatmap
	for i in 0:9
	    for j in 0:9
	        text!(ax3, i, j, text = string(Int(confMat[i+1, j+1])), align = (:center, :center), color = :gray, fontsize = 20)
	    end
	end
	Colorbar(fig3[1, 2], hm)
	fig3
end

# ╔═╡ d11a35ea-8460-4776-8d09-fd42a6971767
md"
As you can see, the current model managed to classify the correct class most of the time without any apparent systematic errors. However, only $(round(tr(confMat) / sum(confMat)*100; digits=2))% of the test set was correctly classified, so there is definitely plenty of room for improvement. The ratio of correctly classified instances to the total number of instances is commonly known as **accuracy**.
"

# ╔═╡ 41d016a5-00a0-432e-8ae6-996709bdd757
accuracy = tr(confMat) / sum(confMat)

# ╔═╡ 8e799e8c-065e-4c28-b597-4c6dab62b3d0
md"""
### Data Augmentation
In modern deep learning, the quality and quantity of training data often matter more than tweaks to model architecture. On MNIST—a 60 000‑sample training set with 10 000 test images—error rates have plunged below 0.3%, with simple CNN ensembles achieving up to 99.91% accuracy, suggesting that further gains hinge on more (or more varied) data rather than new network designs . When gathering new data is costly, it’s it’s essential to maximize the use of existing data.

A common approach is data augmentation, where you apply label‑preserving transformations (e.g. small rotations, translations, noise or elastic distortions) to synthesize additional training examples, boosting robustness and effectively enlarging your dataset without fresh data collection.
"""

# ╔═╡ 28fdfe13-3872-4dc6-8e6d-e7a20400a807
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

# ╔═╡ f70dd72a-6dca-4ee2-9af1-4b324c59c67b
md"""
###  Generalization
It's important for the classifier to perform well not only on the test set, but also on the type of data it will encounter when deployed. For example, consider a digit that we wrote ourselves. Suddenly, the model does not perform as expected. This kind of behavior can result from a distributional shift between the training data and the data the model is presented with during deployment.
"""

# ╔═╡ 5807261e-c5e0-4d50-8cc0-353583eab644
im0 = reshape(Float32.(load("testImage.png")) |> transpose, 28,28,1,:);

# ╔═╡ 05e31b2a-0427-44a8-a4f0-85fb903f7cc8
begin
	fig5 = Figure(size = (450, 450))
	
	# Create an axis for the heatmap
	ax5 = Axis(fig5[1, 1], 
	          title = "Label = 1, Prediction = $(onecold(model(im0), 0:9)[1])", )
	hidedecorations!(ax5)
	heatmap!(ax5,im0[:,end:-1:1,1,1], colormap = :grays, colorrange = (0, 1))
	fig5
end

# ╔═╡ 31a2b5aa-3891-4be8-bf00-34bf1532ba5c
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
# ╟─4656940a-c9c7-11eb-186d-8bd3fedf80e2
# ╠═793b0fa9-28f5-4916-9732-6d953d9aa22f
# ╟─6b6e78ce-4ac8-4caf-8a37-5ea74a124830
# ╠═00b0d98c-375a-4e6c-9a1b-fab2d53bfd18
# ╠═5be0a82d-14e7-4a3d-b1ac-cf25a7b70093
# ╟─abc054bc-2928-4088-8bfe-346724e6e36a
# ╟─080667aa-0c5e-4c9a-89f8-90489b114d19
# ╟─cde625f9-f6ca-48d7-ae31-4ec93bda748b
# ╟─89968d90-d65d-43c7-ba89-95da468628c5
# ╠═4cad4ec3-08d4-4b57-ba49-4a9a63d89aff
# ╟─7be6f10f-8a32-4cca-922d-198e5e4efd71
# ╟─94666008-bfa9-4b50-a84b-aa7d92e1a93f
# ╠═363f8093-4960-4c67-a444-fba097de2a13
# ╟─13f609b2-c986-4494-bb08-72f5903c395e
# ╠═4b9c3e33-43e4-45f1-b8bc-63c0236e9179
# ╟─a24a2469-9081-459e-8382-5c8ea0d4a82e
# ╠═d1ee8eed-5a36-4050-a733-3455b16bb833
# ╠═9eb390a8-6dbb-45de-80b0-fb5f072e66c9
# ╟─24223983-08b4-4966-a3aa-812b6fa2ccda
# ╟─92d96f31-0292-483b-9581-77306624ef28
# ╠═a8ca0f30-e936-45b2-9aad-8259a1ba9a72
# ╟─133db6fb-e7f8-4fa7-ad84-22a4fb5c818b
# ╠═efc55b09-af32-430b-9429-ddfdcfde2f3f
# ╟─d5fc3650-21da-418a-ba33-422d4b33b625
# ╟─d9cb5a95-e424-4f61-9aa8-643d5d5ceab0
# ╠═9b36e662-52de-4c5f-b4ca-00d58399f072
# ╟─c7b47cfa-71a3-4df4-b86f-d0699365982e
# ╟─d11a35ea-8460-4776-8d09-fd42a6971767
# ╠═41d016a5-00a0-432e-8ae6-996709bdd757
# ╟─8e799e8c-065e-4c28-b597-4c6dab62b3d0
# ╟─28fdfe13-3872-4dc6-8e6d-e7a20400a807
# ╟─f70dd72a-6dca-4ee2-9af1-4b324c59c67b
# ╠═5807261e-c5e0-4d50-8cc0-353583eab644
# ╠═05e31b2a-0427-44a8-a4f0-85fb903f7cc8
# ╟─31a2b5aa-3891-4be8-bf00-34bf1532ba5c
