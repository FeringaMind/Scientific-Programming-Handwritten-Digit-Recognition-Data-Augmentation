using Test
using Flux
using GLMakie                 
using GLMakie.Makie              
include("../src/model_backend.jl")
using .LeNet5                    

# Helper function to generate a small dataset for fast tests
function get_dummy_data(pptt=100)
    xtrain, ytrain = LeNet5.getData_train(pptt=pptt)
    xtest, ytest = LeNet5.getData_test()
    return xtrain, ytrain, xtest, ytest
end

@testset "createModel" begin
    model = LeNet5.createModel()

    @test model isa Chain                        # Model should be a Flux.Chain
    @test length(model) == 9                     # LeNet5 has 9 layers (Conv + Pool + Flatten + Dense + Softmax)

    dummy_input = rand(Float32, 28, 28, 1, 1)     # Create dummy MNIST image
    output = model(dummy_input)

    @test size(output) == (10, 1)                 # Output should be for 10 digit classes
    @test all(output .>= 0)                       # Softmax output must be ≥ 0
    @test isapprox(sum(output), 1.0f0; atol=1e-5) # Softmax should sum to ~1
end

@testset "getData_train (pptt)" begin
    xtrain, ytrain = LeNet5.getData_train(pptt=100)  # Use 1% of the dataset

    @test size(xtrain)[1:3] == (28, 28, 1)           # Image shape check
    @test size(ytrain, 1) == 10                      # One-hot encoding: 10 rows
    @test size(xtrain, 4) == size(ytrain, 2)         # Number of images matches labels
    @test all(sum(ytrain, dims=1) .== 1)             # One-hot encoding validation
end

@testset "getData_train (amounts)" begin
    xtrain, ytrain = LeNet5.getData_train(amounts=fill(5, 10))  # 5 samples per class, 50 total

    @test size(xtrain, 4) == 50
    @test size(xtrain)[1:3] == (28, 28, 1)
    @test size(ytrain, 1) == 10
    @test size(ytrain, 2) == 50
    @test all(sum(ytrain, dims=1) .== 1)
end

@testset "getData_test" begin
    xtest, ytest = LeNet5.getData_test()

    @test size(xtest)[1:3] == (28, 28, 1)
    @test size(ytest, 1) == 10
    @test size(xtest, 4) == size(ytest, 2)
    @test all(sum(ytest, dims=1) .== 1)
end

@testset "makeFigurePluto_Images" begin
    x, y = LeNet5.getData_train(pptt=10)                # Load small dataset
    fig = LeNet5.makeFigurePluto_Images(800, 400, x[:, :, :, 1:5], y[:, 1:5])

    @test fig isa GLMakie.Makie.Figure                
end

# Test confusion matrix visualization
@testset "makeFigurePlutoConfusionMatrix" begin
    _, _, xtest, ytest = get_dummy_data()
    preds = Flux.onecold(ytest[:, 1:100], 0:9)           # test labels as predictions
    true_labels = Flux.onecold(ytest[:, 1:100], 0:9)

    fig = LeNet5.makeFigurePluto_ConfusionMatrix(preds, true_labels)
    @test fig isa GLMakie.Makie.Figure
end

@testset "train!" begin
    model = LeNet5.createModel()
    xtrain, ytrain = LeNet5.getData_train(pptt=100)  # small dataset for speed

    no_aug = (x, y) -> (x, y)  # no augmentation
    dict = Dict(model => no_aug)

    loss_history = LeNet5.train!(dict, (xtrain, ytrain); epochs=2, batchsize=32)

    @test all(x -> x isa Real, loss_history)

    # There should be multiple loss entries and ideally a decrease over time
    @test length(loss_history) >= 2
    @test loss_history[end] <= loss_history[1]

end

@testset "test" begin
    model = LeNet5.createModel()
    _, _, xtest, ytest = get_dummy_data()

    preds = LeNet5.test(model, (xtest, ytest))

    @test preds isa Vector{Int}
    @test length(preds) == size(xtest, 4)
    @test all(preds .>= 0)
    @test all(preds .<= 9)
end

@testset "overall_accuracy" begin
    preds = [0, 1, 2, 3, 4]
    labels = [0, 1, 9, 3, 0]                             
    acc = LeNet5.overall_accuracy(preds, labels)

    @test isapprox(acc, 60.0; atol=1e-5)
end

@testset "accuracy_per_class" begin
    preds = [0, 1, 2, 2, 0, 9, 9]
    labels = [0, 1, 1, 2, 0, 9, 8]                    

    acc_map = LeNet5.accuracy_per_class(preds, labels)

    @test acc_map isa Dict
    @test length(acc_map) == 10
    @test acc_map[0][1] == 100.0
    @test acc_map[1][1] == 50.0
    @test acc_map[2][1] == 100.0
end