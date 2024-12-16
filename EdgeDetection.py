import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection_with_grid(image_path, grid_size=(10, 10), low_threshold=100, high_threshold=200):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cant read: {image_path}")

    # Canny
    edges = cv2.Canny(image, low_threshold, high_threshold)


    height, width = edges.shape

 
    grid_height, grid_width = grid_size
    cell_height = height // grid_height
    cell_width = width // grid_width

    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

   
    for i in range(1, grid_height):
        y = i * cell_height
        cv2.line(color_edges, (0, y), (width, y), (0, 255, 0), 1)
    for j in range(1, grid_width):
        x = j * cell_width
        cv2.line(color_edges, (x, 0), (x, height), (0, 255, 0), 1)


    for i in range(grid_height):
        for j in range(grid_width):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width
            cell = edges[y_start:y_end, x_start:x_end]
            edge_count = np.sum(cell) // 255
            
            cv2.putText(color_edges, str(edge_count), (x_start + cell_width // 2, y_start + cell_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

 
    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(color_edges, cv2.COLOR_BGR2RGB))
    plt.title('Edge Image with Grid'), plt.xticks([]), plt.yticks([])
    plt.show()

    return color_edges

# 
image_path = 'xxxxx.jpg'  # 
edges_with_grid = edge_detection_with_grid(image_path)
