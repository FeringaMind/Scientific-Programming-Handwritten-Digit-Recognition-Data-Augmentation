module LeNet5
    ### Usings & Imports
    using Flux, MLDatasets, GLMakie

    GLMakie.activate!(inline=false)

    ### Functions
    """
    createModel()

    Creates and returns a LeNet5-style convolutional neural network using Flux.
    The model consists of convolutional, pooling, dense layers, and a final softmax.

    Returns:
        A `Chain` model from Flux representing the neural network.
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

        print("LeNet5: Created a model with $(sum(length,Flux.trainables(model))) parameters\n")

        return model
    end

    """
    getData()

    Returns the training and test data with labels and downloads the datasets if not already present

    Returns:
        A `xtrain, ytrain, xtest, ytest` where x is the data and y represents the labels.
    """
    function getData()
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

        print("Loaded $(train_size[end]) train images á $(train_size[1])x$(train_size[2])x$(train_size[3]) w/ labels\n")
        print("Loaded $(test_size[end]) test images á $(test_size[1])x$(test_size[2])x$(test_size[3]) w/ labels\n")

        return xtrain, ytrain, xtest, ytest
    end

    function showImages(x_set, y_set)
        ## ToDO
    end

    ### Exports
    export createModel
    export getData
    export showImages
end

### Main Test

model = LeNet5.createModel();
xtrain, ytrain, xtest, ytest = LeNet5.getData();
fig_train = LeNet5.showImages(xtrain, ytrain)
fig_test = LeNet5.showImages(xtest,ytest)

display(fig_train)
nothing