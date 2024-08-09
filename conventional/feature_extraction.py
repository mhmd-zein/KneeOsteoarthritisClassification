import numpy as np
import numpy as np
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm
from skimage.filters import gabor
from conventional.transforms import transforms
from data.dataset import get_loader
from configs import config 

def crop_center(images, masks, margin = 30):
    def find_indices(arr, target):
        min_index = None
        max_index = None
        for i in range(len(arr)):
            if (arr[i] == target).any():
                if min_index is None:
                    min_index = i
                max_index = i
        return min_index, max_index
    images = (images.tolist())
    masks = (masks.tolist())
    for mask_idx in range(len(masks)):
        mask = masks[mask_idx][0].copy()
        unique_values_per_row = [np.unique(row) for row in mask]
        min2, max2 = find_indices(unique_values_per_row, 2)
        min3, max3 = find_indices(unique_values_per_row, 3)
        line = ((min2+max2+min3+max3)-len(mask))/2
        if line>224-margin-1:
            line = 224-margin-1
        if line<margin+1:
            line = margin+1
        masks[mask_idx][0] = mask[int(line-margin):int(line+margin)]
        images[mask_idx][0] = images[mask_idx][0][int(line-margin):int(line+margin)]
    return np.asarray(images), np.asarray(masks)

def calculate_distance(masks):
    masks = np.asarray(masks)
    masks = masks[:,0,:,:]
    black_pixels_count = np.sum(masks == 0, axis=(-2))
    distances = black_pixels_count.reshape(-1, masks.shape[2])
    return distances 

def center_mean(images, masks):
    images, masks = np.asarray(images), np.asarray(masks)
    center_means = np.mean(images*(masks>0), axis=(1,2,3))
    return center_means

def horizontal_similarity(images, masks):
    images = np.asarray(images)
    masks = np.asarray(masks)
    images_right = images[:,:,:,:images.shape[-1]//2]
    images_left = images[:,:,:,images.shape[-1]//2:]
    masks_right = masks[:,:,:,:masks.shape[-1]//2]
    masks_left = masks[:,:,:,masks.shape[-1]//2:]
    images_right_sum = np.sum(images_right, axis=(1,2))
    images_left_sum = np.flip(np.sum(images_left, axis=(1,2)), axis=1)
    masks_right_sum = np.sum(masks_right, axis=(1,2))
    masks_left_sum = np.flip(np.sum(masks_left, axis=(1,2)), axis=1)
    images_corr = np.asarray([np.corrcoef(images_left_sum_i, images_right_sum_i)[0,1] for images_left_sum_i, images_right_sum_i in zip(images_left_sum, images_right_sum)])
    masks_corr = np.asarray([np.corrcoef(masks_left_sum_i, masks_right_sum_i)[0,1] for masks_left_sum_i, masks_right_sum_i in zip(masks_left_sum, masks_right_sum)])
    return images_corr, masks_corr

def calculate_lbp(images):
    batch_features = []
    for i in range(images.shape[0]):
        image = images[i, 0] 
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        batch_features.append(hist)
    return np.array(batch_features)

def calculate_hog(images):
    images = np.asarray(images)
    batch_features = []
    for i in range(images.shape[0]):
        image = images[i, 0] 
        features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=False)
        batch_features.append(features)
    return np.array(batch_features)

def calculate_glcm(images):
    batch_features = []
    for i in range(images.shape[0]):
        image = (images[i, 0]*255).astype(np.uint8) 
        glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)
        features = []
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in props:
            feature = graycoprops(glcm, prop)
            features.extend(feature.flatten())
        batch_features.append(features)
    return np.array(batch_features)

def calculate_distance_features(distances):
    features = np.zeros((distances.shape[0], 7))
    for i, distance in enumerate(distances):
        features[i, 0] = np.std(distance)
        features[i, 1] = np.max(distance) - np.min(distance)
        q75, q25 = np.percentile(distance, [75 ,25])
        features[i, 2] = q75 - q25
        features[i, 3] = entropy(distance)
        features[i, 4] = skew(distance)
        features[i, 5] = kurtosis(distance)
    return features

def calculate_gabor(images):
    images = np.asarray(images)
    gabor_features = []
    frequencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    for image in images:
        image_features = []
        for frequency in frequencies:
            filt_real, filt_imag = gabor(image[0], frequency)
            image_features.append(filt_real.mean())
            image_features.append(filt_real.var())
            image_features.append(filt_imag.mean())
            image_features.append(filt_imag.var())
        gabor_features.append(image_features)
    return np.array(gabor_features)

dataset_path = config.data['path']
train_loader = get_loader(dataset_path, transforms=transforms, mode='train', balance=False, shuffle=config.data['shuffle'], batch_size=config.data['batch_size'], seed=config.seed)
val_loader = get_loader(dataset_path, transforms=transforms, mode='val', shuffle=False, batch_size=config.data['batch_size'])
test_loader = get_loader(dataset_path, transforms=transforms, mode='test', shuffle=False, batch_size=config.data['batch_size'])

feature_vectors = []
loaders_classes = []
for loader in [train_loader, val_loader, test_loader]:
  distances_all = []
  center_means_all = []
  classes_all = []
  image_corr_all = []
  mask_corr_all = []
  hog_all = []
  glcm_all = []
  lbp_all = []
  distance_features_all = []
  gabor_all = []
  for i, batch in enumerate(tqdm(loader)):
      image = batch['image'].numpy()
      mask = batch['mask'].numpy()
      mask = np.where(mask == 30, 0, np.where(mask == 137, 2, np.where(mask == 215, 3, mask))).astype(np.uint8)
      classes = batch['label'].numpy()
      image_cropped, mask_cropped = crop_center(image, mask, margin=30)
      distances = calculate_distance(mask_cropped)
      distance_features = calculate_distance_features(distances)
      center_means = center_mean(image_cropped, mask_cropped)
      image_corr, mask_corr = horizontal_similarity(image_cropped, mask_cropped)
      hog_vector = calculate_hog(image_cropped)
      glcm = calculate_glcm(image_cropped)
      lbp = calculate_lbp(image_cropped)
      gabor_feature = calculate_gabor(image_cropped)
      lbp_all.append(lbp)
      hog_all.append(hog_vector)
      glcm_all.append(glcm)
      image_corr_all.append(image_corr)
      mask_corr_all.append(mask_corr)
      center_means_all.append(center_means)
      distances_all.append(distances)
      distance_features_all.append(distance_features)
      gabor_all.append(gabor_feature)
      classes_all.append(classes)
  distances_all = np.concatenate(distances_all, 0)
  distance_features_all = np.concatenate(distance_features_all, 0)
  center_means_all = np.concatenate(center_means_all, 0)
  mask_corr_all = np.concatenate(mask_corr_all, 0)
  image_corr_all = np.concatenate(image_corr_all, 0)
  glcm_all = np.concatenate(glcm_all, 0)
  lbp_all = np.concatenate(lbp_all, 0)
  hog_all = np.concatenate(hog_all, 0)
  gabor_all = np.concatenate(gabor_all, 0)
  classes_all = np.concatenate(classes_all, 0)

  feature_vectors.append(np.concatenate([
      distances_all, distance_features_all, np.expand_dims(center_means_all, 1),
      np.expand_dims(image_corr_all, 1), np.expand_dims(mask_corr_all, 1),
      glcm_all, lbp_all, hog_all, gabor_all
  ], axis=1))
  loaders_classes.append(classes_all)
  
train_features, val_features, test_features = feature_vectors
train_classes, val_classes, test_classes = loaders_classes