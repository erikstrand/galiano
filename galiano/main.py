import numpy as np
import laspy

with laspy.open('data/bc_092b084_3_4_2_xyes_8_utm10_2019.laz') as fh:
    print('Points from Header:', fh.header.point_count)
    las = fh.read()
    print(las)
    print('Points from data:', len(las.points))

    point_format = las.point_format
    print("point format")
    print(point_format)
    print(point_format.id)
    print(list(point_format.dimension_names))
    print("")

    x = las.X
    y = las.Y
    z = las.Z
    print(x.shape, y.shape, z.shape)

    ground_pts = las.classification == 2
    bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)
    print('Ground Point Return Number distribution:')
    for r,c in zip(bins,counts):
        print('    {}:{}'.format(r,c))
