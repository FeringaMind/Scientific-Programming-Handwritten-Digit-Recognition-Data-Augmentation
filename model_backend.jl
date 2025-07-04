module LeNet5
    ### Usings & Imports
    using Flux, MLDatasets, CairoMakie, InteractiveUtils, Statistics

    ### Functions
    """
    createModel()

    Creates and returns a LeNet5-style convolutional neural network using Flux.
    The model consists of convolutional, pooling, dense layers, and a final softmax.

    Returns:
        model: A `Chain` model from Flux representing the neural network.
    """
    function createModel()
        model = Chain(
            Conv((5, 5), 1=>6, relu),   # C1 Convolution Layer:             28x28x1   =>   24x24x6
            MaxPool((2,2)),             # S2 Pooling Layer:                 24x24x6   =>   12x12x6
            Conv((5, 5), 6=>16, relu),  # C3 Convolution Layer:             12x12x6   =>   8x8x16
            MaxPool((2,2)),             # S4 Pooling Layer:                 8x8x16    =>   4x4x16
            Flux.flatten,               # Flatten the 'volume':             4x4x16    =>   256x1x1
            Dense(256, 120, relu),      # C5/F5 Convolution/Dense Layer:    256x1x1   =>   120x1x1
            Dense(120, 84, relu),       # F6 Dense layer:                   120x1x1   =>   84x1x1
            Dense(84, 10),              # F7 (Output) Dense layer:          84x1x1    =>   10x1x1
            softmax                     # Softmax to create a probability distribution
        )

        println("LeNet5: Created a model with $(sum(length,Flux.trainables(model))) parameters\n")

        return model
    end

    """
    getData(percent)

    Gets the training and test data with labels and downloads the datasets if not already present

    Takes:
        percentage: percentage of training data to return. The test data stays at its original size (default=1)

    Returns:
        xtrain, ytrain, xtest, ytest: x... is the data and y... represents the labels.
    """
    function getData(percentage=1)
        # Disable manual confirmation of dataset download
        ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

        # Loading Dataset	
        xtrain, ytrain = MLDatasets.MNIST(Float32, dir="mnist_data", split=:train)[:]
        xtest, ytest = MLDatasets.MNIST(Float32, dir="mnist_data", split=:test)[:]
        
        # Reshape Data in order to flatten each image into a linear array
        xtrain = reshape(xtrain, 28,28,1,:)
        xtest = reshape(xtest, 28,28,1,:)

        # One-hot-encode the labels
        ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

        train_size = size(xtrain)
        test_size = size(xtest)

        println("Loaded $(train_size[end]) train images á $(train_size[1])x$(train_size[2])x$(train_size[3]) w/ labels")
        println("Loaded $(test_size[end]) test images á $(test_size[1])x$(test_size[2])x$(test_size[3]) w/ labels\n")

        return xtrain, ytrain, xtest, ytest
    end

    """
    makeFigurePluto_Images(x_size, y_size, x_set, y_set, plotslice)

    Creates a figure w/ CairoMakie displayable in a Pluto Notebook

    Takes:
        x_size: size of the figure in x-direction
        y_size: size of the figure in y-direction
        x_set: the image set
        y_set: the label set
        plotslice: the slider input

    Returns:
        fig: The figure
    """
    function makeFigurePluto_Images(x_size, y_size, x_set, y_set, plotslice)
        indices = 12 * (plotslice - 1) + 1 : 12 * plotslice
	    fig = Figure(size = (x_size, y_size), fontsize=20)
	    for (i, idx) in enumerate(indices)
	        ax = Axis(fig[(i-1)÷4+1, (i-1)%4+1], title = "label=$(onecold(y_set[:, idx], 0:9))")
		    hidedecorations!(ax)
	        heatmap!(ax, reshape(x_set, 28,28,1,:)[:,end:-1:1,1,idx], colormap = :grays, colorrange = (0, 1))
	    end
	    return fig
    end

    """
    makeFigurePluto_ConfusionMatrix(y_hat,y)

    #todo 

    Takes:
        y_hat:
        y:
        x_size: size of the figure in x-direction
        y_size: size of the figure in y-direction

    Returns:
        fig: the figure

    """

    function makeFigurePluto_ConfusionMatrix(y_hat, y; x_size=600, y_size=600)

        confMat = zeros(Float64, (10,10))
        for i=0:9, j=0:9
            confMat[i+1,j+1] = sum((y_hat .== i) .&& (y .== j))
        end
        
        confMat = confusionMatrix(preds, onecold(ytest, 0:9))

        # Create a figure
        fig = Figure(size = (x_size, y_size))
        
        # Create an axis for the heatmap
        ax = Axis(fig[1, 1], 
                title = "Confusion Matrix", 
                xlabel = "predicted digit", 
                ylabel = "true digit", 
                xticks = 0:9, 
                yticks = 0:9)
        
        # Plot the heatmap
        hm = heatmap!(ax, 0:9, 0:9, confMat, colormap = :viridis)
        
        # Annotate the heatmap
        for i in 0:9
            for j in 0:9
                text!(ax, i, j, text = string(Int(confMat[i+1, j+1])), align = (:center, :center), color = :gray, fontsize = 20)
            end
        end
        Colorbar(fig[1, 2], hm)

        return fig
    end


    """
    train!(model, data; epochs=10, batchsize=32, lambda=1e-2, eta=3e-4)

    Trains the model with the input parameters

    Takes:
        model: The model to be trained (created by createModel())
        data: The data to train the model with (created by getData())
        epochs: The number of epochs to train (default=10)
        batchsize: The batchsize to be used during training (default=32)
        lambda: The weight decay (default=1e-2)
        eta: The learning rate (default=3e-4)

    Returns:
        loss_history: The loss_history of the training
    """
    function train!(model, data; epochs=10, batchsize=32, lambda=1e-2, eta=3e-4)
        # setup data and model
        xtrain, ytrain = data
        train_loader = Flux.DataLoader(data, batchsize=batchsize, shuffle=true)
        loss_function = Flux.crossentropy

        opt_rule = OptimiserChain(WeightDecay(lambda), ADAMW(eta))
        opt_state = Flux.setup(opt_rule, model)

        width = displaysize(stdout)[2]  # columns (width of terminal)
        println("Training for $(epochs) epochs on $(size(xtrain)[end]) images with batchsize $(batchsize) | λ=$(lambda), η=$(eta)")
        
        # perform training
        loss_history = []
        time_history = []
        actual = Flux.onecold(ytrain |> cpu, 0:9)
        for epoch in 1:epochs

            println(repeat("-", width))
            print("Epoch ($(epoch)/$(epochs))...")

            epoch_loss_history = []

            time_train = @elapsed begin
                for xy_cpu in train_loader
                    x, y = xy_cpu |> cpu	# transfer data to device
                    loss, grads = Flux.withgradient(model) do m
                        loss_function(m(x), y)
                    end
                    push!(loss_history, loss)
                    push!(epoch_loss_history, loss)
                    Flux.update!(opt_state, model, grads[1]) # update parameters pf model
                end
            end
            push!(time_history, time_train)

            y_hat = model(xtrain |> cpu) # get the models prediciton after training
            preds = Flux.onecold(y_hat |> cpu, 0:9)
            correct = count(preds .== actual)
            total = length(actual)
            acc = 100 * correct / total

            println("trained for $(round(time_train, digits=3))s | reached $(round(mean(epoch_loss_history), digits=3)) loss with $(round(acc, digits=3))% accuracy")
        end

        y_hat = model(xtrain |> cpu) # get the models prediciton after training
        preds = Flux.onecold(y_hat |> cpu, 0:9)
        correct = count(preds .== actual)
        total = length(actual)
        acc = 100 * correct / total

        println(repeat("-", width))
        println("Trained for a total of $(round(sum(time_history), digits=3))s | reached a $(round(loss_history[end], digits=3)) loss with $(round(acc, digits=3))% accuracy\n")
        
        return loss_history
    end


    """
    test(model, data,)

    tests the model, the model classifies test data

    Takes:
        model: The model to be trained (created by createModel())
        data: The data to train the model with (created by getData())

    Returns:
        predictions: the class predictions of the model
    """
    function test(model, data;)

        (xtest, ytest) = data

        #yhat is the output of the model (probability distribution)
        y_hat = model(xtest |> cpu)

        # onecold turns the probability distribution of the model into the number classes
        preds = Flux.onecold( y_hat |> cpu, 0:9)

        return preds
    end

    """
    accuracy()
    #todo
    """
    # todo function accuracy(preds, )

    ### Exports
    export createModel
    export getData
    export makeFigurePluto_Images
    export makeFigurePluto_ConfusionMatrix
    export train!
    export test
end