from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename: str) -> np.ndarray:
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset: np.ndarray) -> np.ndarray:
    return np.dot(np.transpose(dataset), dataset) / (len(dataset[0]) - 1)

def get_eig(S: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    eigen_values, eigen_vectors = eigh(S)
    # make every row refer to a eigenvector
    eigen_vectors = np.transpose(eigen_vectors)
    sort_pair = sorted(zip(eigen_values, eigen_vectors), key=lambda item: item[0], reverse=True)
    eigen_values, eigen_vectors = zip(*sort_pair)
    # top m
    eigen_values = np.diag(eigen_values[:m])
    eigen_vectors = np.transpose(eigen_vectors[:m])
    return eigen_values, eigen_vectors

def get_eig_prop(S: np.ndarray, prop: float) -> tuple[np.ndarray, np.ndarray]:
    eigen_values, eigen_vectors = eigh(S)
    # make every row refer to a eigenvector
    eigen_vectors = np.transpose(eigen_vectors)
    sort_pair = sorted(zip(eigen_values, eigen_vectors), key=lambda item: item[0], reverse=True)
    eigen_values, eigen_vectors = zip(*sort_pair)
    # count eigenvalues which are greater than prop
    total = np.sum(eigen_values)
    for idx, value in enumerate(eigen_values):
        if value / total < prop:
            break
    eigen_values = np.diag(eigen_values[:idx])
    eigen_vectors = np.transpose(eigen_vectors[:idx])
    return eigen_values, eigen_vectors


def project_image(image: np.ndarray, U: np.ndarray) -> np.ndarray:
    return np.dot(U, np.dot(np.transpose(U), image))   

def display_image(orig: np.ndarray, proj: np.ndarray) -> tuple:
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    orig = np.reshape(orig, (64, 64))
    proj = np.reshape(proj, (64, 64))
    ax1.set_title("Original")
    ax2.set_title("Projection")
    ori = ax1.imshow(orig, aspect='equal')
    fig.colorbar(ori, ax=ax1)
    proj = ax2.imshow(proj, aspect='equal')
    fig.colorbar(proj, ax=ax2)
    # plt.show()
    return fig, ax1, ax2

def display_image_combo(orig1: np.ndarray, orig2: np.ndarray, comb_proj: np.ndarray) -> tuple:
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    orig1 = np.reshape(orig1, (64, 64))
    orig2 = np.reshape(orig2, (64, 64))
    comb_proj = np.reshape(comb_proj, (64, 64))
    ax1.set_title("Original1")
    ax2.set_title("Original2")
    ax3.set_title("Combined Projection")
    ori1 = ax1.imshow(orig1, aspect='equal')
    fig.colorbar(ori1, ax=ax1)
    ori2 = ax2.imshow(orig2, aspect='equal')
    fig.colorbar(ori2, ax=ax2)
    comb_proj = ax3.imshow(comb_proj, aspect='equal')
    fig.colorbar(comb_proj, ax=ax3)
    # plt.show()
    return fig, ax1, ax2, ax3

def perturb_image(image: np.ndarray, U: np.ndarray, sigma: float) -> np.ndarray:
    proj = project_image(image, U)
    perturb = np.random.normal(loc=0, scale=sigma, size=len(proj))
    proj = [a + b for a, b in zip(proj, perturb)]
    return np.array(proj)

def combine_image(image1: np.ndarray, image2: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    combined_image = lam * project_image(image1, U) + (1-lam) * project_image(image2, U)
    return combined_image

# if __name__ == "__main__":
#     filename = "face_dataset.npy"
#     dataset = load_and_center_dataset(filename)
#     covariance = get_covariance(dataset)
#     k = 100
#     sigma = 1000
#     prob = 0.07
#     lam = 0.5
#     # eigen_values, eigen_vectors = get_eig(covariance, k)
#     eigen_values, eigen_vectors = get_eig_prop(covariance, prob)
#     projected_image = project_image(dataset[50], eigen_vectors)
#     fig, ax1, ax2 = display_image(dataset[50], projected_image)
#     perturbed_image = perturb_image(dataset[50], eigen_vectors, sigma)
#     fig, ax1, ax2 = display_image(dataset[50], perturbed_image)
#     combined_image = combine_image(dataset[50], dataset[80], eigen_vectors, lam)
#     fig, ax1, ax2, ax3 = display_image_combo(dataset[50], dataset[80], combined_image)