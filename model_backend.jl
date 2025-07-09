module LeNet5
    ### Usings & Imports
    using Flux, MLDatasets, CairoMakie, InteractiveUtils, Statistics, Random

    import Flux

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
    getData_train(;pptt, amounts)

    Gets the training data with labels and downloads the datasets if not already present
    --below the main function there are two helper functions for calling with pptt or amounts--

    Takes:
        pptt: pptt of training data to return. The test data stays at its original size (default=1)
        amounts: An array of the amount of representation for each label

    Returns:
        xtrain, ytrain: x... is the data and y... represents the labels.
    """
    function getData_train(;pptt=10000, amounts=missing)

        if ismissing(amounts)
            return getData_train_pptt(pptt=pptt)
        else
            return getData_train_amounts(amounts=amounts)
        end
    end


    function getData_train_pptt(;pptt=10000)
        # Disable manual confirmation of dataset download
        ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

        # Loading Dataset	
        xtrain_raw, ytrain_raw = MLDatasets.MNIST(Float32, dir="mnist_data", split=:train)[:]

        #Initializing the data matrices
        count = 1
        xtrain = zeros(Float32, 28, 28, ceil(Int, 60000*(pptt/10000)))
        ytrain = zeros(Float32, ceil(Int, 60000*(pptt/10000)))

        actual_amount = 0

        #Filtering train data: Images per label
        for label in 0:9
            inds = findall(ytrain_raw .== label)
            inds = Random.shuffle(inds) #shuffling for a randomized training set

            for i in inds[1:(floor(Int,(pptt/10000)*(length(inds))))] 
                xtrain[:,:,count] = xtrain_raw[:,:,i]
                ytrain[count] = ytrain_raw[i]
                count += 1
            end
            actual_amount += length(1:floor(Int,(pptt/10000)*(length(inds))))
        end

        #Cutting the data matrices to "actual amount"
        xtrain = xtrain[:, :, 1:end-(ceil(Int, 60000*(pptt/10000))-(actual_amount))]
        ytrain = ytrain[1:end-(ceil(Int, 60000*(pptt/10000))-(actual_amount))]
        
        # Reshape Data in order to flatten each image into a linear array
        xtrain = reshape(xtrain, 28,28,1,:)

        # One-hot-encode the labels
        ytrain = Flux.onehotbatch(ytrain, 0:9)

        return xtrain, ytrain

    end

    function getData_train_amounts(;amounts)
        # Disable manual confirmation of dataset download
        ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" 

         # Loading Dataset	
        xtrain_raw, ytrain_raw = MLDatasets.MNIST(Float32, dir="mnist_data", split=:train)[:]

        count = 1
        actual_amount = sum(amounts)
        

        xtrain = zeros(Float32, 28, 28, actual_amount)
        ytrain = zeros(Float32, actual_amount)

        for label in 0:9
            inds = findall(ytrain_raw .== label)
            inds = Random.shuffle(inds) #shuffling for a randomized training set

            for i in inds[1:amounts[label+1]] 
                xtrain[:,:,count] = xtrain_raw[:,:,i]
                ytrain[count] = ytrain_raw[i]
                count += 1
            end
        end

        # Reshape Data in order to flatten each image into a linear array
        xtrain = reshape(xtrain, 28,28,1,:)

        # One-hot-encode the labels
        ytrain = Flux.onehotbatch(ytrain, 0:9)

        return xtrain, ytrain
    end

     """
    getData_test()

    Gets the test data with labels and downloads the datasets if not already present

    Takes: (no input)

    Returns:
        xtest, ytest: x... is the data and y... represents the labels.
    """


    function getData_test()
        ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

        # Loading Dataset
        xtest, ytest = MLDatasets.MNIST(Float32, dir="mnist_data", split=:test)[:]

        # Reshape Data in order to flatten each image into a linear array
        xtest = reshape(xtest, 28,28,1,:)

        # One-hot-encode the labels
        ytest = Flux.onehotbatch(ytest, 0:9)

        return xtest, ytest
    end

    """
    makeFigurePluto_Images(x_size, y_size, x_set, y_set)

    Creates a figure w/ CairoMakie displayable in a Pluto Notebook

    Takes:
        x_size: size of the figure in x-direction
        y_size: size of the figure in y-direction
        x_set: the image set
        y_set: the label set

    Returns:
        fig: The figure
    """
    function makeFigurePluto_Images(x_size, y_size, x_set, y_set)
	    fig = Figure(size = (x_size, y_size), fontsize=8)
	    for i in 1:size(y_set)[2]
	        ax = Axis(fig[floor(Int, (i-1)÷10)+1, floor(Int, (i-1)%10)+1], title = "label=$(Flux.onecold(y_set[:, i], 0:9))")
		    hidedecorations!(ax)
	        heatmap!(ax, reshape(x_set, 28,28,1,:)[:,end:-1:1,1,i], colormap = :grays, colorrange = (0, 1))
	    end
	    return fig
    end

    """
    makeFigurePluto_ConfusionMatrix(y_hat,y)

    Creates a figure representing a confusion Matrix w/ CairoMakie displayable in a Pluto Notebook

    Takes:
        y_hat: Predicted class labels (vector of Ints)
        y: True class labels (vector of Ints)
        x_size: Size of the figure in x-direction
        y_size: Size of the figure in y-direction

    Returns:
        fig: The figure
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
    train!(model, data; epochs=10, batchsize=32, lambda=1e-2, eta=3e-4, chance=0.1, aug_fun::Function= (x, y) -> (x, 0))

    Trains the model with the input parameters (with or without augmentations)

    Takes:
        model: The model to be trained (created by createModel())
        data: The data to train the model with (created by getData())
        epochs: The number of epochs to train (default=10)
        batchsize: The batchsize to be used during training (default=32)
        lambda: The weight decay (default=1e-2)
        eta: The learning rate (default=3e-4)
        chance: The chance of using an augmentation on an image
        aug_fun: An augmentation function

    Returns:
        loss_history: The loss_history of the training
    """

    function train!(model, data; epochs=10, batchsize=32, lambda=1e-2, eta=3e-4, chance=0.1, aug_fun::Function= (x, y) -> (x, 0))

        # setup data and model
        xtrain, ytrain = data

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
    
            #Augmentation with aug_fun in each epoch

            aug_data, amount_augmented = aug_fun(data, chance)

            train_loader = Flux.DataLoader(aug_data, batchsize=batchsize, shuffle=true)

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

            println("trained for $(round(time_train, digits=3))s | reached $(round(mean(epoch_loss_history), digits=3)) loss with $(round(acc, digits=3))% accuracy | $(amount_augmented) augmented")
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
    accuracy_per_class(preds, labels)

    Calculates the classification accuracy per digit.

    Takes:
        preds: Predicted class labels (vector of Ints)
        labels: True class labels (vector of Ints)

    Returns:
        Dict mapping digit => accuracy
    """
    function accuracy_per_class(preds::Vector{Int}, labels::Vector{Int})
        class_accuracies = Dict{Int, Tuple{Float64, Int}}() # empty dictionary to store accuracy per digit
        for digit in 0:9
            mask = labels .== digit # boolean mask where the true label matches the current digit
            total = count(mask) # Count the times this digit appears in the true labels
            correct = count(preds[mask] .== digit)  # count correct predictions
            class_accuracies[digit] = ((total == 0 ? 0.0 : correct / total * 100), total) # adding 0.0 accuracy if digit not in data, else compute pptt accuracy
        end
        return class_accuracies
    end

    """
    overall_accuracy(preds, labels)

    Calculates the total classification accuracy (in %).

    Takes:
        preds: predicted class labels (vector of Ints)
        labels: true class labels (vector of Ints)

    Returns:
        accuracy: overall accuracy as a Float64 pptt
    """
    function overall_accuracy(preds::Vector{Int}, labels::Vector{Int})
        correct = count(preds .== labels) # count number of correct predictions
        return correct / length(labels) * 100 # compute pptt of accuracy of correct predictions
    end
    
    ### Exports
    export createModel
    export getData_train
    export getData_test
    export makeFigurePluto_Images
    export makeFigurePluto_ConfusionMatrix
    export train!
    export test
    export overall_accuracy
    export accuracy_per_class
end