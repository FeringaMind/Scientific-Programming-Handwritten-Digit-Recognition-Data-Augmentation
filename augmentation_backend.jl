module Augmentation
    ### USINGS
    using Images, ImageTransformations, CoordinateTransformations


    ### FUNCTIONS
    """
    add_noise(image, noise_level)

    Adds noise to an image
    Arguments:
        image: 2D array (grayscale image)
        noise_level: scaling factor for the noise (default = 0.1)
    Returns:
        Augmented image with added noise.
    """
    function add_noise(image, noise_level_range=0.05:0.2)
        noise_level = rand(noise_level_range)
        noise = noise_level * randn(size(image))
        return clamp.(image .+ noise, 0.0, 1.0) # ensures pixel value stays between 0.0 and 1.0
    end

    """
    zoom_image(image, zoom_factor)

    Zooms into the center of the image by a given factor.
    Arguments:
        image: 2D array (grayscale image)
        zoom_factor: scale factor > 1.0 zooms in (default = 1.2)
    Returns:
        Zoomed image.
    """
    function zoom_image(image, zoom_factor_range=0.8:1.2)
        zoom_factor = rand(zoom_factor_range)
        center = Tuple(round.(Int, size(image) ./ 2)) # middle of picture
        tfm = imresize(image, ratio = zoom_factor)
        #recenter(ImageTransformation(zoom_factor, zoom_factor), center) # ScaledTransformation zooms, recenter determines center of image
        #return warp(image, tfm, axes(image), fillvalue=Float32(0.0)) # zoom -> fill empty pixels 
        return tfm
    end

    """
    rotate_image(image, max_angle_deg)

    Rotates the image by a random angle in the range [-max_angle_deg, +max_angle_deg].
    Arguments:
        image: 2D array (grayscale image)
        max_angle_deg: maximum rotation in degrees (default = 20)
    Returns:
        Rotated image.
    """
    function rotate_image(image, max_angle_deg=15) 
        angle_rad = rand(-max_angle_deg:max_angle_deg) * (π / 180) # selects random angle in degrees from the range [–20, +20] and converts it to radians
        center = Tuple(round.(Int, size(image) ./ 2)) # determine center of image
        tfm = recenter(ImageTransformations.Rotations.RotMatrix(angle_rad), center) # rotationmatrix around center
        return warp(image, tfm, axes(image), fillvalue=Float32(0.0)) # rotate -> fill corners black
    end

    """
    apply_augmentation(x_train, y_train; prob, augmentation_fns)

    Applies random data augmentation to a subset of the training data.
    Arguments:
        x_train: 4D array of shape (28,28,1,N), original training images
        y_train: one-hot encoded labels of shape (10,N)
        prob: probability that a given image is augmented (e.g., 0.1 = 10%)
        augmentation_fns: list of augmentation functions to randomly choose from
    Returns:
        x_combined: original and augmented images
        y_combined: corresponding labels
        actual_prob: actual fraction of images that were augmented
    """
    function apply_augmentation_noise(x_train, y_train, prob)

        x_train_aug = deepcopy(x_train)
        n_samples = size(x_train_aug, 4) # Determine the number of training images

        n_augmented = 0
        for i in 1:n_samples
            if rand(0.0:1.0) <= prob # augment with prob precentage
                img = reshape(x_train_aug[:, :, 1, i], (28,28))
                fn = add_noise
                aug_img = fn(img) # Apply the selected function to the image
                aug_img_shp = reshape(aug_img, 28,28,1,1)
                x_train_aug[:, :, 1, i] = aug_img_shp
                n_augmented += 1
            end
        end

        return (x_train_aug, y_train), n_augmented # returns trainingdata, new labels, actual augmentation rate
    end

    function apply_augmentation_rotate(x_train, y_train, prob)

        x_train_aug = deepcopy(x_train)
        n_samples = size(x_train_aug, 4) # Determine the number of training images

        n_augmented = 0
        for i in 1:n_samples
            if rand(0.0:1.0) <= prob # augment with prob precentage
                img = reshape(x_train_aug[:, :, 1, i], (28,28))
                fn = rotate_image
                aug_img = fn(img) # Apply the selected function to the image
                aug_img_shp = reshape(aug_img, 28,28,1,1)
                x_train_aug[:, :, 1, i] = aug_img_shp
                n_augmented += 1
            end
        end

        return (x_train_aug, y_train), n_augmented # returns trainingdata, new labels, actual augmentation rate
    end

    function apply_augmentation_full(x_train, y_train, prob)

        (rot_data_x,rot_data_y), rot_amount = apply_augmentation_rotate(x_train, y_train, prob/4)
        (noise_data_x,noise_data_y), noise_amount = apply_augmentation_noise(rot_data_x, rot_data_y, prob/4)

        return (noise_data_x,noise_data_y), noise_amount+rot_amount # returns trainingdata, new labels, actual augmentation rate
    end

    ### Exports
    export apply_augmentation
    export add_noise
    export zoom_image
    export rotate_image
end
    