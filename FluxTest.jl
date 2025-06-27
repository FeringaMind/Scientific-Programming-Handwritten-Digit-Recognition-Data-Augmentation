### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 8e45e2a2-5327-11f0-1439-ff4a464ab60f
begin
	
	# Install everything, including CUDA, and load packages:
	using Pkg; Pkg.add(["Flux", "CUDA", "cuDNN", "ProgressMeter", "Plots"])
	using Flux, Statistics, ProgressMeter
	using CUDA  # optional
	device = gpu_device()  # function to move data and model to the GPU
	
	# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
	noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
	truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}
	
	# Define our model, a multi-layer perceptron with one hidden layer of size 3:
	model = Chain(
	    Dense(2 => 3, tanh),      # activation function inside layer
	    BatchNorm(3),
	    Dense(3 => 2)) |> device  # move model to GPU, if one is available
	
	# The model encapsulates parameters, randomly initialised. Its initial output is:
	out1 = model(noisy |> device)    # 2×1000 Matrix{Float32}, or CuArray{Float32}
	probs1 = softmax(out1) |> cpu    # normalise to get probabilities (and move off GPU)
	
	# To train the model, we use batches of 64 samples, and one-hot encoding:
	target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
	loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true);
	
	opt_state = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
	
	# Training loop, using the whole data set 1000 times:
	losses = []
	for epoch in 1:1_000
	    for xy_cpu in loader
	        # Unpack batch of data, and move to GPU:
	        x, y = xy_cpu |> device
	        loss, grads = Flux.withgradient(model) do m
	            # Evaluate model and loss inside gradient context:
	            y_hat = m(x)
	            Flux.logitcrossentropy(y_hat, y)
	        end
	        Flux.update!(opt_state, model, grads[1])
	        push!(losses, loss)  # logging, outside gradient context
	    end
	end
	
	opt_state # parameters, momenta and output have all changed
	
	out2 = model(noisy |> device)         # first row is prob. of true, second row p(false)
	probs2 = softmax(out2) |> cpu         # normalise to get probabilities
	mean((probs2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!
	
end

# ╔═╡ 1297fd56-4a5f-4733-bbd3-7c7a2efae777
begin
	using Plots  # to draw the above figure
	
	p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
	p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1))
	p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network", legend=false)
	
	plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))
end

# ╔═╡ Cell order:
# ╠═8e45e2a2-5327-11f0-1439-ff4a464ab60f
# ╠═1297fd56-4a5f-4733-bbd3-7c7a2efae777
