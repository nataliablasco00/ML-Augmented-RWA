import csv

from RWAC import *
from RWANN import *

def test_model(image_path, n_clusters, R, compression_technology, scale, train_split, verbose=False, write_results=False,
               model_path=None, model_mode="RWA", mode="individual_clustering", images_clustering=1):

    image_name = image_path.split('/')[-1]

    if verbose:
        print('Image: ', image_path)

    if "u16be" in image_name:
        dtype = ">u2"
    elif "u16le" in image_name:
        dtype = "<u2"
    elif "s16be" in image_name:
        dtype = ">i2"
    elif "s16le" in image_name:
        dtype = "<i2"
    else:
        raise ValueError("Error: No data type recognized")

    z, y, x = image_name[:-4].split('x')[-3:]
    y = int(y.split('_')[0])
    x = int(x)
    z = int(z.split('-')[-1])


    l = int(np.ceil(np.log2(z)))

    output_folder = f'{"/".join(image_path.split("/")[:-3])}/output/{image_path.split("/")[-2]}_{n_clusters}_{R}_{compression_technology}_{mode}_{images_clustering}_NNClustering'
    rwa_output = output_folder + '_RWA/'
    inv_output = output_folder + '_inv_RWA/'

    os.makedirs(rwa_output, exist_ok=True)
    os.makedirs(inv_output, exist_ok=True)

    rwa_output = rwa_output + image_name[:-4] + '.npy'
    inv_output = inv_output + image_name[:-4] + '.npy'
    
    t_rwa = time.time()
    tr_entropy = RWAC_Transform(image_path, z, y, x, dtype, rwa_output, n_clusters, R, compression_technology, scale,
                                train_split, verbose, mode, model_path=model_path, model_mode=model_mode)
    t_rwa = time.time() - t_rwa
    t_inv = time.time()
    inv_RWAC_Transform(rwa_output, z, y, x, dtype, inv_output, R, compression_technology, scale, model_path=model_path,
                       model_mode=model_mode)
    t_inv = time.time() - t_inv

    or_entropy = entropy(image_path, z=z, y=y, x=x, dtype=dtype)
    
    print('OR ENTROPY: ', str(or_entropy))
    print('TR ENTROPY: ', str(tr_entropy))
    print('RWA TIME: ', str(t_rwa))
    print('INV TIME: ', str(t_inv))

    if write_results:
        with open('model_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([image_name, z, y, x, dtype, rwa_output, n_clusters, R, compression_technology, scale, train_split, t_rwa, t_inv, t_rwa+t_inv, or_entropy, tr_entropy])


    im = np.load(inv_output, allow_pickle=True)

    with open(image_path, 'rb') as f:
        raw_data = f.read()
    image_data = np.frombuffer(raw_data, dtype=dtype)
    true_im = image_data.reshape((z, x * y))
    true_im = true_im.transpose()


    same_recovered = np.array_equal(im, true_im)
    
    if same_recovered:
        print('IDENTICAL')

    else:
        print('DIFFER')
        


    