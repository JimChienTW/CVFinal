import os
import cv2
import random
import face_recognition
import numpy as np


def find_facial_keypoints(face_name, face_path, opt):
    size = opt.face_img_size
    image = face_recognition.load_image_file(face_path)
    face_landmarks_list = face_recognition.face_landmarks(image)

    points = []
    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            points.extend([point for point in face_landmarks[facial_feature]])


    points.extend([(0, 0), (size-1, size-1), (0, size-1), (size-1, 0), 
               (0, int(size/2)), (size-1, int(size/2)), (int(size/2), 0), (int(size/2), size-1)])

    os.makedirs(opt.keypoints_dir, exist_ok=True)
    with open(os.path.join(opt.keypoints_dir, f'{face_name}.txt'), 'w') as f:
        for point in points:
            point = tuple(min(max(value, 0), size-1) for value in point)
            f.write(f'{point[0]} {point[1]}\n')

    return points


def find_tri_pairs(face_name, opt):
    rect = (0, 0, opt.face_img_size, opt.face_img_size)
    subdiv = cv2.Subdiv2D(rect)

    points = []
    with open(os.path.join(opt.keypoints_dir, face_name + '.txt')) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))


    for p in points:
        subdiv.insert(p)

    tris = []
    points = np.array(points).astype(np.int64) # 80*2
    triangle_list = np.array(subdiv.getTriangleList()).reshape((-1, 3, 2)).astype(np.int64) # n*3*2
    for index_t in range(triangle_list.shape[0]):
        comparison = (points[:, np.newaxis, :] == triangle_list[index_t])
        indices = np.where(np.all(comparison, axis=-1))
        unique_value, unique_indices = np.unique(indices[-1], return_index=True)
        
        tri = indices[0][unique_indices]
        tris.append(tri.tolist())
        
        print(indices)  
        
    # save triangle indices
    os.makedirs(opt.tri_dir, exist_ok=True)
    with open(os.path.join(opt.tri_dir, 'tri.txt'), 'w') as f:
        for tri in tris:
            line = [str(index) + ' ' for index in tri]
            f.writelines(line)
            f.write('\n')

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = []
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points    
    

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def change_face(player, alpha=0.5):

    os.makedirs(player.opt.changed_face_dir, exist_ok=True)

    face_source_name = player.name + '.jpg'
    face_target_name = player.target_face + '.jpg'



    face_source_path = os.path.join(player.opt.known_face_dir, face_source_name)
    face_target_path = os.path.join(player.opt.target_face_dir, face_target_name)

    # detect the facial keypoint for source face
    points_source = find_facial_keypoints(face_source_name, face_source_path, player.opt)

    # detect the facial keypoint for target face
    points_target = find_facial_keypoints(face_target_name, face_target_path, player.opt)

    # delaunay triangle
    find_tri_pairs(face_source_name, player.opt)

    # start morphing
    img_source = cv2.imread(face_source_path)
    img_target = cv2.imread(face_target_path)

    img_source = np.float32(img_source)
    img_target = np.float32(img_target)

    points_source = readPoints(os.path.join(player.opt.keypoints_dir, face_source_name + '.txt'))
    points_target = readPoints(os.path.join(player.opt.keypoints_dir, face_target_name + '.txt'))

    # compute weighted average point coordinates
    points = []
    for i in range(0, len(points_source)):
        x = ( 1 - alpha ) * points_source[i][0] + alpha * points_target[i][0]
        y = ( 1 - alpha ) * points_source[i][1] + alpha * points_target[i][1]
        points.append((x,y))

    # initialize the morphed image
    imgMorph = np.zeros(img_source.shape, dtype=img_source.dtype)

    # Read triangle frmo tri.txt
    with open(os.path.join(player.opt.tri_dir, 'tri.txt')) as file:
        for line in file :
            x,y,z = line.split()
            
            x = int(x)
            y = int(y)
            z = int(z)
            
            t1 = [points_source[x], points_source[y], points_source[z]]
            t2 = [points_target[x], points_target[y], points_target[z]]
            t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
            morphTriangle(img_source, img_target, imgMorph, t1, t2, t, alpha)

    cv2.imwrite(os.path.join(player.opt.changed_face_dir, face_source_name), np.uint8(imgMorph))




def change_faces(face_names, opt):
    for name in face_names:
        if name == 'Unknown':
            continue
        # randomly pick one target image
        target_faces_name = [str(file).split('.')[0] for file in os.listdir(opt.target_face_dir) if file.endswith('.jpg')]
        target_face_name = random.choice(target_faces_name)
        # change the face
        change_face(name, target_face_name, opt)