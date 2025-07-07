using Test
using Flux
include("../model_backend.jl")
using .LeNet5  # Import LeNet5 module containing model and utility functions

@testset "createModel" begin
    model = LeNet5.createModel()

    @test model isa Chain # Check that the returned object is a valid Flux model
    @test length(model) == 9 # Verify the expected structure of the model (9 layers including Conv, Pooling, Dense, etc.)
    dummy_input = rand(Float32, 28, 28, 1, 1) # Run a dummy image through the model to check forward pass
    output = model(dummy_input)

    @test size(output) == (10, 1) # Output should be a 10-element vector (one per digit class)

    @test all(output .>= 0) # Ensure the output forms a valid probability distribution
    @test isapprox(sum(output), 1.0f0; atol=1e-5)
end

@testset "getData" begin
    xtrain, ytrain, xtest, ytest = getData()

    # Check training data dimensions: 28x28 grayscale images, 60,000 samples
    @test size(xtrain, 1) == 28
    @test size(xtrain, 2) == 28
    @test size(xtrain, 3) == 1
    @test size(xtrain, 4) == 60000

    # Check test data dimensions: same as training, but 10,000 samples
    @test size(xtest, 4) == 10000

    # Verify one-hot encoded labels for training set: 10 rows (classes), 1 hot per column
    @test size(ytrain, 1) == 10
    @test size(ytrain, 2) == 60000
    @test all(sum(ytrain, dims=1) .== 1)

    # Same checks for test labels
    @test size(ytest, 1) == 10
    @test size(ytest, 2) == 10000
    @test all(sum(ytest, dims=1) .== 1)
end

@testset "train!" begin
    # Initialize model and use only 1% of data for faster testing
    model = LeNet5.createModel()
    xtrain, ytrain, _, _ = getData(0.01)

    # Save model parameters before training to detect changes
    initial_params = deepcopy(Flux.params(model))

    # Train briefly to test functionality and capture training loss
    loss_history = LeNet5.train!(model, (xtrain, ytrain); epochs=1, batchsize=32)

    # Validate that loss values are numeric
    @test all(x -> x isa Real, loss_history)

    # There should be multiple loss entries and ideally a decrease over time
    @test length(loss_history) >= 2
    @test loss_history[end] <= loss_history[1]

    # Ensure that training actually updated model parameters
    has_changed = any(!≈(a, b) for (a, b) in zip(initial_params, Flux.params(model)))
    @test has_changed
end

@testset "accuracy" begin
    model = LeNet5.createModel()
    xtrain, ytrain, xtest, ytest = getData(0.1)
    preds = LeNet5.test(model, (xtest, ytest))
    true_labels = Flux.onecold(ytest, 0:9)
    acc = LeNet5.overall_accuracy(preds, true_labels)
    @test 0.0 <= acc <= 100.0
end

#=
#xtrain, ytrain, xtest, ytest = LeNet5.getData()
#model = LeNet5.createModel()
#loss = LeNet5.train!(model, (xtrain,ytrain);epochs=1)
preds = LeNet5.test(model, (xtrain,ytrain))
@show accT = LeNet5.overall_accuracy(preds, Flux.onecold(ytrain, 0:9))
@show accC = LeNet5.accuracy_per_class(preds, Flux.onecold(ytrain, 0:9))

perc = 1.0
cT = 0
for (key, (perc_val, count)) in accC
    global perc += perc_val * count
    global cT += count
    @show key
    @show perc_per_image = perc_val / count
end
@show round(perc/cT, digits=2)
=#