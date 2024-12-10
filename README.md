# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

# sentiment.txt
```
Epoch 1, loss 31.607361391531715, train accuracy: 48.00%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 2, loss 31.244229469104095, train accuracy: 50.22%
Validation accuracy: 48.00%
Best Valid accuracy: 49.00%
Epoch 3, loss 31.1062771615737, train accuracy: 51.56%
Validation accuracy: 55.00%
Best Valid accuracy: 55.00%
Epoch 4, loss 31.024250331966133, train accuracy: 52.67%
Validation accuracy: 54.00%
Best Valid accuracy: 55.00%
Epoch 5, loss 30.8892287351781, train accuracy: 54.89%
Validation accuracy: 67.00%
Best Valid accuracy: 67.00%
Epoch 6, loss 30.71737733391477, train accuracy: 55.78%
Validation accuracy: 52.00%
Best Valid accuracy: 67.00%
Epoch 7, loss 30.50169279213917, train accuracy: 59.78%
Validation accuracy: 63.00%
Best Valid accuracy: 67.00%
Epoch 8, loss 30.14020640964375, train accuracy: 60.89%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 9, loss 29.995237398120718, train accuracy: 63.56%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 10, loss 29.623043670940206, train accuracy: 65.78%
Validation accuracy: 66.00%
Best Valid accuracy: 68.00%
Epoch 11, loss 29.257109426085606, train accuracy: 67.33%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 12, loss 28.826357346238726, train accuracy: 67.11%
Validation accuracy: 51.00%
Best Valid accuracy: 68.00%
Epoch 13, loss 28.334474414644806, train accuracy: 67.56%
Validation accuracy: 65.00%
Best Valid accuracy: 68.00%
Epoch 14, loss 27.939378040631063, train accuracy: 68.89%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 15, loss 27.3311242802206, train accuracy: 72.44%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 16, loss 27.096261769876794, train accuracy: 70.67%
Validation accuracy: 64.00%
Best Valid accuracy: 71.00%
Epoch 17, loss 26.660645601830762, train accuracy: 70.67%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 18, loss 25.37767935814731, train accuracy: 76.22%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 19, loss 25.623018862288806, train accuracy: 72.89%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 20, loss 24.487924034028353, train accuracy: 73.78%
Validation accuracy: 62.00%
Best Valid accuracy: 75.00%
Epoch 21, loss 23.37665082976689, train accuracy: 74.67%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 22, loss 23.745298655236873, train accuracy: 74.89%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 23, loss 22.368262115040732, train accuracy: 78.00%
Validation accuracy: 74.00%
Best Valid accuracy: 77.00%
Epoch 24, loss 22.48371349448038, train accuracy: 76.00%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 25, loss 21.708573679250854, train accuracy: 77.56%
Validation accuracy: 74.00%
Best Valid accuracy: 77.00%
Epoch 26, loss 20.76299142640014, train accuracy: 78.89%
Validation accuracy: 66.00%
Best Valid accuracy: 77.00%
Epoch 27, loss 19.910642641190755, train accuracy: 80.22%
Validation accuracy: 70.00%
Best Valid accuracy: 77.00%
Epoch 28, loss 20.820152497879487, train accuracy: 77.56%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 29, loss 18.59596780576939, train accuracy: 80.67%
Validation accuracy: 77.00%
Best Valid accuracy: 77.00%
Epoch 30, loss 19.211710468561982, train accuracy: 81.33%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 31, loss 17.944568911690503, train accuracy: 83.78%
Validation accuracy: 74.00%
Best Valid accuracy: 77.00%
Epoch 32, loss 17.631075341995633, train accuracy: 82.00%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 33, loss 17.570498036110532, train accuracy: 82.67%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 34, loss 16.91265190586739, train accuracy: 82.44%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 35, loss 16.88022200187921, train accuracy: 82.22%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 36, loss 16.44811153779495, train accuracy: 80.67%
Validation accuracy: 73.00%
Best Valid accuracy: 77.00%
Epoch 37, loss 15.112675985617026, train accuracy: 86.22%
Validation accuracy: 75.00%
Best Valid accuracy: 77.00%
Epoch 38, loss 15.03503023943126, train accuracy: 84.22%
Validation accuracy: 74.00%
Best Valid accuracy: 77.00%
Epoch 39, loss 14.213463432440383, train accuracy: 88.00%
Validation accuracy: 69.00%
Best Valid accuracy: 77.00%
Epoch 40, loss 13.645072059974506, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 77.00%
Epoch 41, loss 14.360885119488435, train accuracy: 87.78%
Validation accuracy: 78.00%
Best Valid accuracy: 78.00%
Epoch 42, loss 14.206749656737424, train accuracy: 84.89%
Validation accuracy: 69.00%
Best Valid accuracy: 78.00%
Epoch 43, loss 13.996352515013614, train accuracy: 85.78%
Validation accuracy: 76.00%
Best Valid accuracy: 78.00%
Epoch 44, loss 13.723435015503522, train accuracy: 85.33%
Validation accuracy: 77.00%
Best Valid accuracy: 78.00%
Epoch 45, loss 14.41437083913179, train accuracy: 83.78%
Validation accuracy: 69.00%
Best Valid accuracy: 78.00%
Epoch 46, loss 12.734679690316433, train accuracy: 84.67%
Validation accuracy: 69.00%
Best Valid accuracy: 78.00%
Epoch 47, loss 13.292292060644476, train accuracy: 85.78%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 48, loss 12.265831230979726, train accuracy: 86.00%
Validation accuracy: 69.00%
Best Valid accuracy: 78.00%
Epoch 49, loss 11.95475989609639, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 50, loss 12.699855642064062, train accuracy: 84.89%
Validation accuracy: 74.00%
Best Valid accuracy: 78.00%
Epoch 51, loss 12.720834272398738, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 52, loss 12.558506971233587, train accuracy: 84.67%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 53, loss 11.909930947464874, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 54, loss 11.380944914726317, train accuracy: 88.44%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 55, loss 11.126155566907041, train accuracy: 87.78%
Validation accuracy: 68.00%
Best Valid accuracy: 78.00%
Epoch 56, loss 11.893004773966734, train accuracy: 85.11%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 57, loss 11.240709425566314, train accuracy: 86.44%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 58, loss 11.93960574242389, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 59, loss 10.243595941364928, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 60, loss 10.946382834605762, train accuracy: 87.56%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 61, loss 10.33302392635293, train accuracy: 89.33%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 62, loss 10.993882051854612, train accuracy: 84.89%
Validation accuracy: 74.00%
Best Valid accuracy: 78.00%
Epoch 63, loss 11.279134498934482, train accuracy: 84.67%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 64, loss 9.627813548761845, train accuracy: 87.78%
Validation accuracy: 69.00%
Best Valid accuracy: 78.00%
Epoch 65, loss 11.225385661590854, train accuracy: 85.11%
Validation accuracy: 74.00%
Best Valid accuracy: 78.00%
Epoch 66, loss 10.64636098151317, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 67, loss 9.846453471988164, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 68, loss 9.898776755144263, train accuracy: 88.89%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 69, loss 9.295994069888819, train accuracy: 90.67%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 70, loss 10.679499232260294, train accuracy: 84.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 71, loss 9.390021224260432, train accuracy: 88.00%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 72, loss 10.02813611748068, train accuracy: 87.33%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 73, loss 9.785917999730678, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 74, loss 10.22093569310146, train accuracy: 86.67%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 75, loss 8.318322968732712, train accuracy: 89.78%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 76, loss 10.204642277637822, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 77, loss 9.856091144772384, train accuracy: 86.89%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 78, loss 10.229699383821178, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 79, loss 9.330824706807471, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 80, loss 10.188443825024567, train accuracy: 85.78%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 81, loss 9.897189468625067, train accuracy: 87.11%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 82, loss 10.737027019899504, train accuracy: 84.44%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 83, loss 8.943990886358188, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 84, loss 9.456434454846836, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 85, loss 9.36977485227354, train accuracy: 85.78%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 86, loss 9.462340948541145, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 87, loss 8.20832362093525, train accuracy: 89.11%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 88, loss 9.930426436150627, train accuracy: 83.33%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 89, loss 9.477432894316209, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 90, loss 7.88007028988967, train accuracy: 90.22%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 91, loss 10.295580284618966, train accuracy: 85.78%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 92, loss 9.230448789009943, train accuracy: 86.44%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 93, loss 9.740803996263596, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 94, loss 8.749737708112574, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 95, loss 9.625675627247402, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 96, loss 9.910919223176414, train accuracy: 83.56%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 97, loss 8.200060116389755, train accuracy: 89.11%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 98, loss 9.875401765157367, train accuracy: 84.67%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 99, loss 9.166128727094618, train accuracy: 86.22%
Validation accuracy: 70.00%
Best Valid accuracy: 78.00%
Epoch 100, loss 8.473041698335816, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 101, loss 9.113699249359488, train accuracy: 86.67%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 102, loss 9.78198297640674, train accuracy: 83.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 103, loss 10.11691334849607, train accuracy: 84.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 104, loss 8.666503392660742, train accuracy: 90.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 105, loss 8.200449745489182, train accuracy: 90.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 106, loss 8.31755762645821, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 107, loss 8.498271129585758, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 108, loss 8.950681672248592, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 109, loss 9.500475195063045, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 110, loss 8.853234083659094, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 111, loss 8.525589450248695, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 112, loss 9.35440665031966, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 113, loss 8.199416313238393, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 114, loss 7.849612055597679, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 115, loss 8.456540338619995, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 116, loss 8.563277436090717, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 117, loss 8.874778050186244, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 118, loss 8.647489849103984, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 119, loss 8.964507197584231, train accuracy: 86.89%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 120, loss 9.406393779266425, train accuracy: 84.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 121, loss 8.385867246812314, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 122, loss 8.93218125164749, train accuracy: 84.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 123, loss 7.976601321385901, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 124, loss 8.630918525798586, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 125, loss 8.895997030623821, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 126, loss 8.608590409064519, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 127, loss 8.657798392554861, train accuracy: 87.56%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 128, loss 9.155664728834292, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 129, loss 8.888598357276397, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 130, loss 7.4411859369971145, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 131, loss 8.43335771463357, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 132, loss 7.682768953294518, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 133, loss 7.580145889770084, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 134, loss 8.163438602623701, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 135, loss 8.538852825072194, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 136, loss 8.513158991872281, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 137, loss 7.40492230699252, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 138, loss 8.278036955981829, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 139, loss 9.666192924388236, train accuracy: 84.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 140, loss 7.4180109544130115, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 141, loss 8.267426466293822, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 142, loss 9.305601369663922, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 143, loss 9.12144964565098, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 144, loss 8.779955448220807, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 145, loss 8.648024361668453, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 146, loss 8.84064716887529, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 147, loss 8.551225433938662, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 148, loss 8.152345795726589, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 149, loss 8.39776176522474, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 150, loss 8.310112222890117, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 151, loss 8.426415600236925, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 152, loss 8.582320467148156, train accuracy: 89.11%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 153, loss 8.079522296998995, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 154, loss 9.210919434133258, train accuracy: 84.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 155, loss 8.220548583355063, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 156, loss 6.654137588808686, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 157, loss 8.305227906570416, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 158, loss 8.060620040920076, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 159, loss 10.006869108548754, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 160, loss 8.486658465709287, train accuracy: 88.22%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 161, loss 9.012472397409052, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 162, loss 7.946404808012208, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 163, loss 7.916502381932422, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 164, loss 9.070725937157707, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 165, loss 8.894033293443671, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 166, loss 8.257675918516375, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 167, loss 8.275895331892206, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 168, loss 8.002200887226957, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 169, loss 8.370394352248445, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 170, loss 8.429147877602286, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 171, loss 8.853965056299256, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 172, loss 8.76246895939969, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 173, loss 9.602421742364033, train accuracy: 83.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 174, loss 6.911088172783353, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 175, loss 9.426290894538322, train accuracy: 86.22%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 176, loss 7.35312892056301, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 177, loss 8.713681984583381, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 178, loss 7.306355052078696, train accuracy: 88.22%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 179, loss 8.473388918850732, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 180, loss 7.7300763952530085, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 181, loss 7.4684342343526815, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 182, loss 9.241576873588345, train accuracy: 87.11%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 183, loss 9.07190357974912, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 184, loss 9.57644319040651, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 185, loss 8.834443885683893, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 186, loss 7.994163973825327, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 187, loss 8.438963316332453, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 188, loss 8.874870887870266, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 189, loss 8.715757404021794, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 190, loss 8.310535851946625, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 191, loss 6.88887084195416, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 192, loss 8.9295198417019, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 193, loss 7.965381160702316, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 194, loss 8.855998457213715, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 195, loss 8.818811415662264, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 196, loss 9.722533155749803, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 197, loss 8.446547791180977, train accuracy: 84.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 198, loss 7.9965195531648625, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 199, loss 7.522606238931469, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 200, loss 8.830996180274648, train accuracy: 82.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 201, loss 7.8585646117171315, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 202, loss 7.203188644970417, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 203, loss 8.350818455890034, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 204, loss 8.467459488683698, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 205, loss 9.438796274906851, train accuracy: 83.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 206, loss 7.989919393492021, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 207, loss 7.818596315718405, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 208, loss 8.731781574412349, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 209, loss 7.968909053169959, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 210, loss 9.306785209813372, train accuracy: 85.56%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 211, loss 7.480317111548784, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 212, loss 7.92668942279641, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 213, loss 7.575079501680347, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 214, loss 8.668507152827695, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 215, loss 7.90688571421088, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 216, loss 9.399986161454695, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 217, loss 7.4145029379906084, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 218, loss 7.694767407539832, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 219, loss 7.274014835845968, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 220, loss 9.661098885642378, train accuracy: 83.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 221, loss 8.875423680794475, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 222, loss 8.527844555870814, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 223, loss 7.901066191304202, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 224, loss 8.875662166722835, train accuracy: 84.22%
Validation accuracy: 74.00%
Best Valid accuracy: 78.00%
Epoch 225, loss 8.573214895730407, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 226, loss 8.096776645988882, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 227, loss 8.80158866097692, train accuracy: 84.67%
Validation accuracy: 74.00%
Best Valid accuracy: 78.00%
Epoch 228, loss 8.18407264036761, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 229, loss 8.770914023625096, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 230, loss 6.662008748336839, train accuracy: 90.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 231, loss 7.074085538101637, train accuracy: 89.33%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 232, loss 7.256800152310818, train accuracy: 90.22%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 233, loss 6.699366543855302, train accuracy: 89.33%
Validation accuracy: 74.00%
Best Valid accuracy: 78.00%
Epoch 234, loss 8.425968396292594, train accuracy: 84.22%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 235, loss 8.128020052724795, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 236, loss 8.635612943049017, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 237, loss 9.171647003290042, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 238, loss 8.222927047712517, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 239, loss 8.554839794580472, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 240, loss 7.594211547933286, train accuracy: 88.22%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 241, loss 8.542376798734118, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 242, loss 7.7971410949673325, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 243, loss 8.738025119270246, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 244, loss 8.689923639468464, train accuracy: 86.22%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 245, loss 8.335822930745044, train accuracy: 84.44%
Validation accuracy: 73.00%
Best Valid accuracy: 78.00%
Epoch 246, loss 8.756953701121008, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 247, loss 8.556165530624215, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 248, loss 8.116571572871855, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
Epoch 249, loss 8.453632760902812, train accuracy: 87.56%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 250, loss 7.170129534009556, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
```
# mnist.txt
```
Epoch 1 loss 2.3015274934597802 valid acc 2/16
Epoch 1 loss 11.484321837588467 valid acc 1/16
Epoch 1 loss 11.490236394511385 valid acc 1/16
Epoch 1 loss 11.484840096587353 valid acc 3/16
Epoch 1 loss 11.453353448462698 valid acc 4/16
Epoch 1 loss 11.436819150790758 valid acc 3/16
Epoch 1 loss 11.26125365303643 valid acc 5/16
Epoch 1 loss 11.154651374190713 valid acc 7/16
Epoch 1 loss 11.10079565749195 valid acc 8/16
Epoch 1 loss 9.94505638808661 valid acc 8/16
Epoch 1 loss 8.978355565149082 valid acc 11/16
Epoch 1 loss 8.125290433062135 valid acc 12/16
Epoch 1 loss 7.187551094627707 valid acc 15/16
Epoch 1 loss 6.2598649763582905 valid acc 13/16
Epoch 1 loss 6.653937276632433 valid acc 13/16
Epoch 1 loss 5.331057931842941 valid acc 9/16
Epoch 1 loss 6.617347945105658 valid acc 12/16
Epoch 1 loss 6.136600403969287 valid acc 10/16
Epoch 1 loss 5.771223559988604 valid acc 12/16
Epoch 1 loss 4.548737716594578 valid acc 13/16
Epoch 1 loss 4.4622698264730944 valid acc 12/16
Epoch 1 loss 4.998836855711301 valid acc 13/16
Epoch 1 loss 3.1138372068349227 valid acc 13/16
Epoch 1 loss 5.352380304806111 valid acc 15/16
Epoch 1 loss 3.7484066273642185 valid acc 12/16
Epoch 1 loss 4.466640468792795 valid acc 15/16
Epoch 1 loss 3.743706860848847 valid acc 14/16
Epoch 1 loss 3.2730839147217914 valid acc 14/16
Epoch 1 loss 3.7760679135016195 valid acc 14/16
Epoch 1 loss 3.532721401463503 valid acc 14/16
Epoch 1 loss 4.424936485748079 valid acc 11/16
Epoch 1 loss 4.058506050342343 valid acc 13/16
Epoch 1 loss 2.8258789993075677 valid acc 12/16
Epoch 1 loss 4.2889988880605925 valid acc 15/16
Epoch 1 loss 6.371243332461521 valid acc 14/16
Epoch 1 loss 4.33877965708945 valid acc 14/16
Epoch 1 loss 2.6832962967187077 valid acc 11/16
Epoch 1 loss 3.6841351513656453 valid acc 13/16
Epoch 1 loss 4.258747349196687 valid acc 15/16
Epoch 1 loss 3.7383177152648894 valid acc 14/16
Epoch 1 loss 3.348163844846275 valid acc 15/16
Epoch 1 loss 3.767586242206779 valid acc 16/16
Epoch 1 loss 4.110395458177863 valid acc 12/16
Epoch 1 loss 3.451522406906263 valid acc 15/16
Epoch 1 loss 3.6783902473796957 valid acc 16/16
Epoch 1 loss 3.133219480805427 valid acc 15/16
Epoch 1 loss 3.300672671460488 valid acc 15/16
Epoch 1 loss 2.9904485922183754 valid acc 16/16
Epoch 1 loss 3.086001552071108 valid acc 13/16
Epoch 1 loss 3.2291583914948196 valid acc 16/16
Epoch 1 loss 3.4135391120129968 valid acc 14/16
Epoch 1 loss 5.439842626756399 valid acc 14/16
Epoch 1 loss 4.41964835965976 valid acc 15/16
Epoch 1 loss 4.278221316769587 valid acc 12/16
Epoch 1 loss 3.102892220613872 valid acc 14/16
Epoch 1 loss 2.787149539865446 valid acc 14/16
Epoch 1 loss 2.927564410152317 valid acc 14/16
Epoch 1 loss 2.410789342119719 valid acc 15/16
Epoch 1 loss 3.0957738771743823 valid acc 15/16
Epoch 1 loss 3.570128823304758 valid acc 15/16
Epoch 1 loss 3.0369539719746963 valid acc 13/16
Epoch 1 loss 3.1803961781738024 valid acc 13/16
Epoch 1 loss 3.9464869544492713 valid acc 14/16
Epoch 2 loss 0.20266133375461715 valid acc 14/16
Epoch 2 loss 4.646503877643138 valid acc 15/16
Epoch 2 loss 3.6932873827023065 valid acc 15/16
Epoch 2 loss 3.527997885879465 valid acc 14/16
Epoch 2 loss 2.3398588163826384 valid acc 14/16
Epoch 2 loss 2.7522608015415524 valid acc 13/16
Epoch 2 loss 3.3969663894277913 valid acc 14/16
Epoch 2 loss 3.432360071409691 valid acc 14/16
Epoch 2 loss 4.016243333122242 valid acc 14/16
Epoch 2 loss 3.085190616268619 valid acc 15/16
Epoch 2 loss 2.8552980752340766 valid acc 15/16
Epoch 2 loss 4.833235171055202 valid acc 15/16
Epoch 2 loss 4.213845785411363 valid acc 15/16
Epoch 2 loss 3.1221559598976443 valid acc 15/16
Epoch 2 loss 5.499046953481933 valid acc 14/16
Epoch 2 loss 2.139413232925839 valid acc 14/16
Epoch 2 loss 4.204332219950527 valid acc 14/16
Epoch 2 loss 5.886665122561635 valid acc 9/16
Epoch 2 loss 4.352348653883995 valid acc 15/16
Epoch 2 loss 3.092619180273955 valid acc 15/16
Epoch 2 loss 2.5767088404780036 valid acc 15/16
Epoch 2 loss 2.86191387075665 valid acc 15/16
Epoch 2 loss 1.8985271268455 valid acc 15/16
Epoch 2 loss 3.0779228525193156 valid acc 15/16
Epoch 2 loss 2.7523665962593507 valid acc 13/16
Epoch 2 loss 2.561324851741016 valid acc 14/16
Epoch 2 loss 4.641533601766331 valid acc 13/16
Epoch 2 loss 3.1625139376789955 valid acc 15/16
Epoch 2 loss 2.816919841943833 valid acc 15/16
Epoch 2 loss 1.6179199111975515 valid acc 15/16
Epoch 2 loss 2.3638380263044425 valid acc 14/16
Epoch 2 loss 2.7036424010414803 valid acc 14/16
Epoch 2 loss 1.8800486710711983 valid acc 13/16
Epoch 2 loss 2.209922951656379 valid acc 15/16
Epoch 2 loss 6.007561797644399 valid acc 16/16
Epoch 2 loss 3.454733264691696 valid acc 14/16
Epoch 2 loss 2.368660501715313 valid acc 15/16
Epoch 2 loss 2.102847335716972 valid acc 15/16
Epoch 2 loss 2.8142763712484653 valid acc 15/16
Epoch 2 loss 2.380030645585224 valid acc 14/16
Epoch 2 loss 2.2672542041041877 valid acc 14/16
Epoch 2 loss 2.98301028658425 valid acc 16/16
Epoch 2 loss 2.3735088513018803 valid acc 15/16
Epoch 2 loss 2.173008524353962 valid acc 15/16
Epoch 2 loss 3.6180665740344518 valid acc 13/16
Epoch 2 loss 2.854651135464815 valid acc 15/16
Epoch 2 loss 2.4288836731058088 valid acc 15/16
Epoch 2 loss 2.7991875871392677 valid acc 15/16
Epoch 2 loss 3.065220919587445 valid acc 15/16
Epoch 2 loss 1.9685836990406658 valid acc 15/16
Epoch 2 loss 2.3320728105129716 valid acc 15/16
Epoch 2 loss 2.7902044883272463 valid acc 15/16
Epoch 2 loss 2.8681249482313116 valid acc 15/16
Epoch 2 loss 2.4378265463715225 valid acc 15/16
Epoch 2 loss 3.176094049945588 valid acc 15/16
Epoch 2 loss 2.3260868538894 valid acc 15/16
Epoch 2 loss 2.3716610825116167 valid acc 15/16
Epoch 2 loss 1.713730853862692 valid acc 15/16
Epoch 2 loss 2.214470848182241 valid acc 13/16
Epoch 2 loss 3.4730727031571442 valid acc 15/16
Epoch 2 loss 2.8091408376506033 valid acc 14/16
Epoch 2 loss 1.8852093344510674 valid acc 15/16
Epoch 2 loss 3.276640575802404 valid acc 15/16
Epoch 3 loss 0.25284636937263233 valid acc 15/16
Epoch 3 loss 2.664033421444507 valid acc 15/16
Epoch 3 loss 3.503967359244365 valid acc 15/16
Epoch 3 loss 2.4723409083719705 valid acc 15/16
Epoch 3 loss 1.7959119205163967 valid acc 15/16
Epoch 3 loss 2.384569636814894 valid acc 15/16
Epoch 3 loss 2.0435120886027267 valid acc 15/16
Epoch 3 loss 2.7133935597634276 valid acc 15/16
Epoch 3 loss 2.7987470610900678 valid acc 15/16
Epoch 3 loss 2.415010883076137 valid acc 14/16
Epoch 3 loss 1.9136289190882687 valid acc 16/16
Epoch 3 loss 4.351805127638146 valid acc 15/16
Epoch 3 loss 2.5439961330974743 valid acc 15/16
Epoch 3 loss 2.997645388125016 valid acc 15/16
Epoch 3 loss 3.32615616310073 valid acc 15/16
Epoch 3 loss 1.7401113181001628 valid acc 15/16
Epoch 3 loss 4.670735242465058 valid acc 15/16
Epoch 3 loss 2.6479983326167194 valid acc 14/16
Epoch 3 loss 2.775415271631396 valid acc 15/16
Epoch 3 loss 2.473693271133595 valid acc 15/16
Epoch 3 loss 2.179062256345391 valid acc 16/16
Epoch 3 loss 2.3258615638136906 valid acc 15/16
Epoch 3 loss 0.8393659043585416 valid acc 16/16
Epoch 3 loss 1.5683786725130924 valid acc 15/16
Epoch 3 loss 1.4150006654128584 valid acc 15/16
Epoch 3 loss 1.520894169360759 valid acc 15/16
Epoch 3 loss 2.73227276227069 valid acc 15/16
Epoch 3 loss 1.4377361549677372 valid acc 15/16
Epoch 3 loss 1.5438824885481142 valid acc 15/16
Epoch 3 loss 1.4482168960363857 valid acc 15/16
Epoch 3 loss 1.8690152605425283 valid acc 14/16
Epoch 3 loss 2.707293149670156 valid acc 14/16
Epoch 3 loss 0.9544455175316231 valid acc 15/16
Epoch 3 loss 1.998800351294805 valid acc 15/16
Epoch 3 loss 4.614565855020057 valid acc 15/16
Epoch 3 loss 3.322037365600486 valid acc 15/16
Epoch 3 loss 2.7915005252230403 valid acc 16/16
Epoch 3 loss 2.675724983710252 valid acc 15/16
Epoch 3 loss 1.9528793361014407 valid acc 15/16
Epoch 3 loss 2.3004519162676953 valid acc 15/16
Epoch 3 loss 1.3693945956017648 valid acc 16/16
Epoch 3 loss 3.1436969583478795 valid acc 16/16
Epoch 3 loss 2.5090388303498314 valid acc 15/16
Epoch 3 loss 1.7267884003556195 valid acc 15/16
Epoch 3 loss 2.2117352446713063 valid acc 13/16
Epoch 3 loss 1.904167440278396 valid acc 15/16
Epoch 3 loss 2.672658465394793 valid acc 11/16
Epoch 3 loss 4.900123934617344 valid acc 12/16
Epoch 3 loss 3.410749222430341 valid acc 15/16
Epoch 3 loss 1.8229799949877277 valid acc 15/16
Epoch 3 loss 2.337048896190169 valid acc 13/16
Epoch 3 loss 3.1489590865695067 valid acc 15/16
Epoch 3 loss 2.6104643357815 valid acc 15/16
Epoch 3 loss 1.7216100421316316 valid acc 15/16
Epoch 3 loss 2.25862549820678 valid acc 15/16
Epoch 3 loss 2.7098496801992953 valid acc 15/16
Epoch 3 loss 2.608570487834601 valid acc 15/16
Epoch 3 loss 2.415984883250118 valid acc 15/16
Epoch 3 loss 4.199280810228158 valid acc 11/16
Epoch 3 loss 4.688563033467928 valid acc 13/16
Epoch 3 loss 5.118293476998817 valid acc 13/16
Epoch 3 loss 4.363109121960366 valid acc 13/16
Epoch 3 loss 5.574857885093488 valid acc 14/16
Epoch 4 loss 0.8058868021630242 valid acc 14/16
Epoch 4 loss 4.802613134324603 valid acc 13/16
Epoch 4 loss 4.680825713304148 valid acc 12/16
Epoch 4 loss 4.2014919347286215 valid acc 12/16
Epoch 4 loss 2.606153628428156 valid acc 15/16
Epoch 4 loss 2.6662890263349928 valid acc 15/16
Epoch 4 loss 2.581916335028738 valid acc 13/16
Epoch 4 loss 3.4246263046630325 valid acc 15/16
Epoch 4 loss 2.6561452082291526 valid acc 15/16
Epoch 4 loss 3.3282680721084574 valid acc 15/16
Epoch 4 loss 2.9170288406399383 valid acc 13/16
Epoch 4 loss 4.750709475255919 valid acc 12/16
Epoch 4 loss 4.272692863866761 valid acc 14/16
Epoch 4 loss 4.0835390169425505 valid acc 14/16
Epoch 4 loss 5.471973201648533 valid acc 13/16
Epoch 4 loss 4.460040377708742 valid acc 16/16
Epoch 4 loss 5.4249497134827624 valid acc 12/16
Epoch 4 loss 4.325147209548555 valid acc 13/16
Epoch 4 loss 2.9116756432303985 valid acc 15/16
Epoch 4 loss 2.3176226366738875 valid acc 13/16
Epoch 4 loss 3.3805981143887642 valid acc 13/16
Epoch 4 loss 2.302078989452634 valid acc 13/16
Epoch 4 loss 1.8747554964756135 valid acc 15/16
Epoch 4 loss 3.985737384411326 valid acc 13/16
Epoch 4 loss 2.427605471117278 valid acc 14/16
Epoch 4 loss 2.5302271576541204 valid acc 14/16
Epoch 4 loss 4.970906711622398 valid acc 15/16
Epoch 4 loss 2.209216070618063 valid acc 15/16
Epoch 4 loss 2.4089855166795555 valid acc 14/16
Epoch 4 loss 1.449700729214809 valid acc 14/16
Epoch 4 loss 2.87575358887291 valid acc 14/16
Epoch 4 loss 2.5337926228276078 valid acc 16/16
Epoch 4 loss 2.486169280585519 valid acc 15/16
Epoch 4 loss 3.0502156597990417 valid acc 16/16
Epoch 4 loss 3.796033565205775 valid acc 15/16
Epoch 4 loss 3.0534165328556635 valid acc 15/16
Epoch 4 loss 3.0792190287032906 valid acc 16/16
Epoch 4 loss 2.6854369509340206 valid acc 16/16
Epoch 4 loss 2.185397934017506 valid acc 15/16
Epoch 4 loss 3.918534352714737 valid acc 14/16
Epoch 4 loss 3.0779093365695704 valid acc 16/16
Epoch 4 loss 2.897833556751299 valid acc 16/16
Epoch 4 loss 2.7520775812466773 valid acc 14/16
Epoch 4 loss 2.4095494344622184 valid acc 15/16
Epoch 4 loss 2.4144039815737743 valid acc 15/16
Epoch 4 loss 1.5479729867052323 valid acc 15/16
Epoch 4 loss 2.6776971508035166 valid acc 13/16
Epoch 4 loss 3.311158252462885 valid acc 15/16
Epoch 4 loss 1.962498671484283 valid acc 16/16
Epoch 4 loss 2.164098194265979 valid acc 15/16
Epoch 4 loss 2.3479302806379354 valid acc 15/16
Epoch 4 loss 2.404801543552245 valid acc 15/16
Epoch 4 loss 2.7035487115411816 valid acc 15/16
Epoch 4 loss 1.583802950817919 valid acc 15/16
Epoch 4 loss 2.525555150658371 valid acc 15/16
Epoch 4 loss 2.8345463455938216 valid acc 14/16
Epoch 4 loss 3.30789380110926 valid acc 15/16
Epoch 4 loss 3.480974872334266 valid acc 14/16
Epoch 4 loss 2.958125727670713 valid acc 13/16
Epoch 4 loss 3.2454125963419553 valid acc 14/16
Epoch 4 loss 2.14133220955589 valid acc 14/16
Epoch 4 loss 2.2716611800565487 valid acc 15/16
Epoch 4 loss 3.6492229841363364 valid acc 14/16
Epoch 5 loss 0.1925982039862631 valid acc 14/16
Epoch 5 loss 3.598857399859646 valid acc 14/16
Epoch 5 loss 3.6603186450771847 valid acc 14/16
Epoch 5 loss 1.9116400876838915 valid acc 14/16
Epoch 5 loss 2.0296873194759586 valid acc 14/16
Epoch 5 loss 1.9253451151822674 valid acc 14/16
Epoch 5 loss 2.1337244868093896 valid acc 13/16
Epoch 5 loss 2.4346201926050526 valid acc 14/16
Epoch 5 loss 3.126836630276936 valid acc 14/16
Epoch 5 loss 1.3540436227985069 valid acc 15/16
Epoch 5 loss 2.1030371899966323 valid acc 15/16
Epoch 5 loss 3.6488413994349482 valid acc 16/16
Epoch 5 loss 2.3745111491060817 valid acc 15/16
Epoch 5 loss 4.226662176088874 valid acc 14/16
Epoch 5 loss 3.325435096726652 valid acc 15/16
Epoch 5 loss 3.60528972181984 valid acc 13/16
Epoch 5 loss 4.0389686078354785 valid acc 15/16
Epoch 5 loss 3.958246347802265 valid acc 14/16
Epoch 5 loss 2.297169565090521 valid acc 15/16
Epoch 5 loss 1.7151863945230548 valid acc 14/16
Epoch 5 loss 2.428832235074631 valid acc 13/16
Epoch 5 loss 1.8221234912621358 valid acc 14/16
Epoch 5 loss 1.20020854172991 valid acc 14/16
Epoch 5 loss 1.9120441919107516 valid acc 14/16
Epoch 5 loss 2.495096830293681 valid acc 14/16
Epoch 5 loss 2.017437554447634 valid acc 14/16
Epoch 5 loss 2.706262138704228 valid acc 14/16
Epoch 5 loss 1.438080748440246 valid acc 16/16
Epoch 5 loss 2.060330189243518 valid acc 15/16
Epoch 5 loss 0.9782242218613257 valid acc 15/16
Epoch 5 loss 2.831200872165695 valid acc 15/16
Epoch 5 loss 2.4720121605125103 valid acc 15/16
Epoch 5 loss 1.3134096680682503 valid acc 16/16
Epoch 5 loss 1.8656331109662172 valid acc 16/16
Epoch 5 loss 3.4998035881203235 valid acc 16/16
Epoch 5 loss 3.144003965505486 valid acc 15/16
Epoch 5 loss 2.3371418504966326 valid acc 14/16
Epoch 5 loss 2.1676255224439904 valid acc 15/16
Epoch 5 loss 1.9401256158537001 valid acc 16/16
Epoch 5 loss 3.034667309015696 valid acc 16/16
Epoch 5 loss 1.871361965733816 valid acc 15/16
Epoch 5 loss 2.485185477402224 valid acc 16/16
Epoch 5 loss 2.5016515933392776 valid acc 15/16
Epoch 5 loss 1.614536886412561 valid acc 14/16
Epoch 5 loss 2.4366545104705324 valid acc 15/16
Epoch 5 loss 1.7109996163019536 valid acc 14/16
Epoch 5 loss 1.5936040081052947 valid acc 15/16
Epoch 5 loss 2.64979198420461 valid acc 14/16
Epoch 5 loss 1.6193934946892765 valid acc 15/16
Epoch 5 loss 1.7417933945017714 valid acc 15/16
Epoch 5 loss 1.5682237530390386 valid acc 15/16
Epoch 5 loss 1.8598875953411016 valid acc 16/16
Epoch 5 loss 2.401083487378963 valid acc 14/16
Epoch 5 loss 1.4271913116909465 valid acc 14/16
Epoch 5 loss 2.6344051442635883 valid acc 14/16
Epoch 5 loss 2.0228001103115707 valid acc 15/16
Epoch 5 loss 1.8939189930039502 valid acc 16/16
Epoch 5 loss 2.667103124103468 valid acc 15/16
Epoch 5 loss 2.2872123338172345 valid acc 13/16
Epoch 5 loss 2.0140702217854605 valid acc 14/16
Epoch 5 loss 2.120887766259516 valid acc 14/16
Epoch 5 loss 2.307283520208361 valid acc 15/16
Epoch 5 loss 2.3921832733100254 valid acc 14/16
Epoch 6 loss 0.2081531875872518 valid acc 14/16
Epoch 6 loss 2.4469667146470324 valid acc 14/16
Epoch 6 loss 2.7897196046417054 valid acc 14/16
Epoch 6 loss 2.1165980335560555 valid acc 14/16
Epoch 6 loss 1.5056047139014057 valid acc 14/16
Epoch 6 loss 2.316043456611119 valid acc 15/16
Epoch 6 loss 1.7434776424598133 valid acc 13/16
Epoch 6 loss 2.4250493185552355 valid acc 14/16
Epoch 6 loss 2.3577342914040327 valid acc 15/16
Epoch 6 loss 1.3371264037402413 valid acc 15/16
Epoch 6 loss 1.651072676753464 valid acc 15/16
Epoch 6 loss 2.829858424764602 valid acc 14/16
Epoch 6 loss 1.9177874994108957 valid acc 14/16
Epoch 6 loss 2.771758996352511 valid acc 14/16
Epoch 6 loss 2.7810585126099463 valid acc 15/16
Epoch 6 loss 2.390569697868033 valid acc 14/16
Epoch 6 loss 3.2291708596186894 valid acc 14/16
Epoch 6 loss 2.213050300496005 valid acc 14/16
Epoch 6 loss 1.7742120191497128 valid acc 14/16
Epoch 6 loss 1.8837242316297318 valid acc 14/16
Epoch 6 loss 2.1928139754178795 valid acc 14/16
Epoch 6 loss 1.4559040721345515 valid acc 13/16
Epoch 6 loss 1.0230595567134522 valid acc 14/16
Epoch 6 loss 1.2816638204952104 valid acc 14/16
Epoch 6 loss 1.3673147722082029 valid acc 14/16
Epoch 6 loss 1.166324859155687 valid acc 15/16
Epoch 6 loss 1.6734151364218828 valid acc 15/16
Epoch 6 loss 1.5888195708478183 valid acc 14/16
Epoch 6 loss 1.9496045266034057 valid acc 14/16
Epoch 6 loss 0.724432208662595 valid acc 15/16
Epoch 6 loss 1.8820439590812152 valid acc 15/16
Epoch 6 loss 1.5878491655570584 valid acc 15/16
Epoch 6 loss 0.5341451056212754 valid acc 15/16
Epoch 6 loss 2.110119502030911 valid acc 15/16
Epoch 6 loss 4.93470014168137 valid acc 15/16
Epoch 6 loss 2.508899414372585 valid acc 14/16
Epoch 6 loss 2.1253528002424424 valid acc 14/16
Epoch 6 loss 1.6360584711230843 valid acc 15/16
Epoch 6 loss 1.9855670228215296 valid acc 15/16
Epoch 6 loss 2.316090928267642 valid acc 15/16
Epoch 6 loss 1.3993878025100377 valid acc 15/16
Epoch 6 loss 1.38795531420671 valid acc 15/16
Epoch 6 loss 2.1615796121532758 valid acc 14/16
Epoch 6 loss 1.8530212261725714 valid acc 14/16
Epoch 6 loss 1.5462234177461454 valid acc 15/16
Epoch 6 loss 1.0616549999204525 valid acc 15/16
Epoch 6 loss 1.806140431919024 valid acc 14/16
Epoch 6 loss 1.9219672686573783 valid acc 15/16
Epoch 6 loss 1.6373289557629844 valid acc 15/16
Epoch 6 loss 1.2336217698700496 valid acc 14/16
Epoch 6 loss 1.6378597960938703 valid acc 14/16
Epoch 6 loss 2.3713454698614145 valid acc 15/16
Epoch 6 loss 2.101464802838854 valid acc 15/16
Epoch 6 loss 1.3434524997563466 valid acc 15/16
Epoch 6 loss 1.592374094397408 valid acc 14/16
Epoch 6 loss 1.5237067836331089 valid acc 15/16
Epoch 6 loss 1.7053453392067965 valid acc 14/16
Epoch 6 loss 2.542777177619808 valid acc 14/16
Epoch 6 loss 2.268836747937108 valid acc 14/16
Epoch 6 loss 2.2515503309979144 valid acc 13/16
Epoch 6 loss 2.267005222378754 valid acc 14/16
Epoch 6 loss 1.669555057362297 valid acc 15/16
Epoch 6 loss 3.0143906168112515 valid acc 14/16
Epoch 7 loss 0.4006011872714618 valid acc 14/16
Epoch 7 loss 2.396334835581761 valid acc 14/16
Epoch 7 loss 2.9801763809934037 valid acc 14/16
Epoch 7 loss 1.7327654783179516 valid acc 14/16
Epoch 7 loss 0.9562712124827071 valid acc 14/16
Epoch 7 loss 1.8207387728922728 valid acc 15/16
Epoch 7 loss 1.7648063481798644 valid acc 14/16
Epoch 7 loss 1.7424386341737879 valid acc 14/16
Epoch 7 loss 1.8730755745102379 valid acc 14/16
Epoch 7 loss 1.0165211232300844 valid acc 15/16
Epoch 7 loss 1.4462231148346465 valid acc 15/16
Epoch 7 loss 2.3407097242385886 valid acc 13/16
Epoch 7 loss 2.266493118669772 valid acc 14/16
Epoch 7 loss 3.0990827788786146 valid acc 15/16
Epoch 7 loss 2.7304342097473207 valid acc 15/16
Epoch 7 loss 1.91840639310213 valid acc 15/16
Epoch 7 loss 3.1312007362417056 valid acc 14/16
Epoch 7 loss 2.149850915229459 valid acc 15/16
Epoch 7 loss 1.6240988537238117 valid acc 14/16
Epoch 7 loss 2.065616358439746 valid acc 14/16
Epoch 7 loss 2.0243859105993236 valid acc 14/16
Epoch 7 loss 0.8888868438822644 valid acc 13/16
Epoch 7 loss 0.860560142752369 valid acc 15/16
Epoch 7 loss 1.5283229012745443 valid acc 14/16
Epoch 7 loss 0.970042813613389 valid acc 14/16
Epoch 7 loss 1.419845041608025 valid acc 14/16
Epoch 7 loss 1.2676972969449196 valid acc 14/16
Epoch 7 loss 0.7102511989030176 valid acc 14/16
Epoch 7 loss 1.2269437391139495 valid acc 14/16
Epoch 7 loss 0.3383078287578854 valid acc 15/16
Epoch 7 loss 1.4230631016312938 valid acc 15/16
Epoch 7 loss 2.0949175908173276 valid acc 15/16
Epoch 7 loss 0.8464635644916697 valid acc 15/16
Epoch 7 loss 1.4073321359853457 valid acc 15/16
Epoch 7 loss 3.028220124632898 valid acc 14/16
Epoch 7 loss 2.1683208448050335 valid acc 14/16
Epoch 7 loss 2.5078927676559863 valid acc 14/16
Epoch 7 loss 1.3835749806825508 valid acc 15/16
Epoch 7 loss 1.062028796640785 valid acc 15/16
Epoch 7 loss 2.4818206391284656 valid acc 15/16
Epoch 7 loss 1.492254151027724 valid acc 15/16
Epoch 7 loss 1.5037118694405842 valid acc 15/16
Epoch 7 loss 1.5527603590725099 valid acc 15/16
Epoch 7 loss 2.321627284910666 valid acc 14/16
Epoch 7 loss 2.253782486624172 valid acc 15/16
Epoch 7 loss 1.3826509970538992 valid acc 14/16
Epoch 7 loss 1.1275756724470554 valid acc 15/16
Epoch 7 loss 2.8874634546681173 valid acc 15/16
Epoch 7 loss 1.2054399850542385 valid acc 15/16
Epoch 7 loss 1.845405695971556 valid acc 15/16
Epoch 7 loss 1.1049996906831672 valid acc 15/16
Epoch 7 loss 2.0689434570518324 valid acc 15/16
Epoch 7 loss 1.7715237574534992 valid acc 15/16
Epoch 7 loss 0.9294048958032606 valid acc 14/16
Epoch 7 loss 2.0914068391304026 valid acc 14/16
Epoch 7 loss 1.5052615480234481 valid acc 14/16
Epoch 7 loss 1.3827902523203535 valid acc 14/16
Epoch 7 loss 2.5179864264726137 valid acc 15/16
Epoch 7 loss 1.9846321796982074 valid acc 14/16
Epoch 7 loss 2.1018500391781334 valid acc 14/16
Epoch 7 loss 1.7461745043705683 valid acc 14/16
Epoch 7 loss 1.074137313881727 valid acc 15/16
Epoch 7 loss 2.325632520627932 valid acc 13/16
Epoch 8 loss 0.1541994315753312 valid acc 14/16
Epoch 8 loss 2.600928525486085 valid acc 14/16
Epoch 8 loss 1.8122746308909323 valid acc 14/16
Epoch 8 loss 1.7905625060603334 valid acc 14/16
Epoch 8 loss 1.140009315819377 valid acc 14/16
Epoch 8 loss 1.2220987167842863 valid acc 15/16
Epoch 8 loss 2.179764423202709 valid acc 14/16
Epoch 8 loss 1.869019016356522 valid acc 14/16
Epoch 8 loss 1.2061492087103305 valid acc 15/16
Epoch 8 loss 1.299650140848688 valid acc 15/16
Epoch 8 loss 0.9050021535748903 valid acc 15/16
Epoch 8 loss 2.7488657820749185 valid acc 14/16
Epoch 8 loss 1.7156546438533948 valid acc 14/16
Epoch 8 loss 2.161165412349386 valid acc 14/16
Epoch 8 loss 2.8108230341482603 valid acc 15/16
Epoch 8 loss 2.005510678604341 valid acc 14/16
Epoch 8 loss 2.6128442333403408 valid acc 14/16
Epoch 8 loss 2.2187197734469226 valid acc 14/16
Epoch 8 loss 2.0955073712548 valid acc 14/16
Epoch 8 loss 2.0465048249321374 valid acc 14/16
Epoch 8 loss 2.3589444932369483 valid acc 14/16
Epoch 8 loss 1.3547629967746526 valid acc 14/16
Epoch 8 loss 1.056895522087139 valid acc 14/16
Epoch 8 loss 1.3071285437698903 valid acc 15/16
Epoch 8 loss 1.2514093706139304 valid acc 14/16
Epoch 8 loss 0.9574036027381041 valid acc 14/16
Epoch 8 loss 1.1694240246266638 valid acc 14/16
Epoch 8 loss 0.9427371120795826 valid acc 14/16
Epoch 8 loss 1.642820452658916 valid acc 14/16
Epoch 8 loss 0.7455935738947259 valid acc 14/16
Epoch 8 loss 1.2141911575353763 valid acc 14/16
Epoch 8 loss 1.6648007045434916 valid acc 15/16
Epoch 8 loss 0.8847688374197443 valid acc 14/16
Epoch 8 loss 1.2023622202746933 valid acc 15/16
Epoch 8 loss 2.428169650064048 valid acc 15/16
Epoch 8 loss 2.310790670215728 valid acc 13/16
Epoch 8 loss 1.3151508595042198 valid acc 14/16
Epoch 8 loss 0.8186439467761192 valid acc 15/16
Epoch 8 loss 1.250741633365256 valid acc 15/16
Epoch 8 loss 1.7277166755411177 valid acc 15/16
Epoch 8 loss 1.2927953225983186 valid acc 15/16
Epoch 8 loss 1.4841905963231172 valid acc 15/16
Epoch 8 loss 1.1654139610388135 valid acc 15/16
Epoch 8 loss 1.7800568126695469 valid acc 14/16
Epoch 8 loss 2.094736774176172 valid acc 15/16
Epoch 8 loss 1.1766581035457508 valid acc 15/16
Epoch 8 loss 1.476647984014799 valid acc 14/16
Epoch 8 loss 1.6125398908874096 valid acc 15/16
Epoch 8 loss 0.9765370117544315 valid acc 15/16
Epoch 8 loss 1.2917079494333998 valid acc 15/16
Epoch 8 loss 0.9509895754142144 valid acc 15/16
Epoch 8 loss 1.875868105551147 valid acc 14/16
Epoch 8 loss 1.404465773698924 valid acc 14/16
Epoch 8 loss 0.7423149670523779 valid acc 15/16
Epoch 8 loss 1.960843021988721 valid acc 14/16
Epoch 8 loss 1.4088731393980236 valid acc 15/16
Epoch 8 loss 1.085725480781128 valid acc 15/16
Epoch 8 loss 2.1144852684371718 valid acc 15/16
Epoch 8 loss 2.105686900523297 valid acc 16/16
Epoch 8 loss 1.4422653646897474 valid acc 14/16
Epoch 8 loss 1.7558384183978375 valid acc 14/16
Epoch 8 loss 1.0146922455961063 valid acc 15/16
Epoch 8 loss 2.127302191470566 valid acc 14/16
Epoch 9 loss 0.17218965942242154 valid acc 14/16
Epoch 9 loss 1.9712714279548342 valid acc 15/16
Epoch 9 loss 1.4378384706679377 valid acc 14/16
Epoch 9 loss 1.837041008052335 valid acc 14/16
Epoch 9 loss 1.0075924588144178 valid acc 14/16
Epoch 9 loss 1.2069908456243859 valid acc 15/16
Epoch 9 loss 2.327679872381288 valid acc 15/16
Epoch 9 loss 1.916637780431984 valid acc 15/16
Epoch 9 loss 1.8371546896203799 valid acc 15/16
Epoch 9 loss 1.0689506729280458 valid acc 15/16
Epoch 9 loss 1.341726113744264 valid acc 15/16
Epoch 9 loss 2.1008696694984987 valid acc 15/16
Epoch 9 loss 1.9426131917913776 valid acc 15/16
Epoch 9 loss 2.26725095633139 valid acc 15/16
Epoch 9 loss 2.078157267829206 valid acc 16/16
Epoch 9 loss 1.6098342610656404 valid acc 15/16
Epoch 9 loss 2.7699148324984257 valid acc 14/16
Epoch 9 loss 1.1743046788164484 valid acc 14/16
Epoch 9 loss 1.6277718379806294 valid acc 15/16
Epoch 9 loss 1.7972214702604443 valid acc 15/16
Epoch 9 loss 1.6427423095708806 valid acc 15/16
Epoch 9 loss 0.6809623883210505 valid acc 15/16
Epoch 9 loss 0.6147480381989631 valid acc 15/16
Epoch 9 loss 1.7085559829835741 valid acc 15/16
Epoch 9 loss 1.099444647294196 valid acc 15/16
Epoch 9 loss 1.1549328135568209 valid acc 15/16
Epoch 9 loss 0.7065686080193972 valid acc 15/16
Epoch 9 loss 1.6065173257205807 valid acc 15/16
Epoch 9 loss 1.1276083920238285 valid acc 15/16
Epoch 9 loss 0.42729993738016486 valid acc 15/16
Epoch 9 loss 1.3634132653215332 valid acc 16/16
Epoch 9 loss 1.9871565204162767 valid acc 15/16
Epoch 9 loss 1.1267303167855958 valid acc 15/16
Epoch 9 loss 1.0667284810656712 valid acc 15/16
Epoch 9 loss 2.2877986814200915 valid acc 15/16
Epoch 9 loss 1.7365736784936323 valid acc 15/16
Epoch 9 loss 1.256215080213929 valid acc 15/16
Epoch 9 loss 0.7516268891758734 valid acc 15/16
Epoch 9 loss 1.309212500573948 valid acc 15/16
Epoch 9 loss 1.8919845889920461 valid acc 15/16
Epoch 9 loss 1.2928410156994667 valid acc 15/16
Epoch 9 loss 1.260603506043696 valid acc 15/16
Epoch 9 loss 1.525602085576639 valid acc 15/16
Epoch 9 loss 1.5582767989620074 valid acc 15/16
Epoch 9 loss 1.836792680479148 valid acc 15/16
Epoch 9 loss 1.2208223326532623 valid acc 15/16
Epoch 9 loss 1.1351356880862902 valid acc 15/16
Epoch 9 loss 1.6686720194924307 valid acc 15/16
Epoch 9 loss 1.1230797070516203 valid acc 15/16
Epoch 9 loss 1.186225280180248 valid acc 15/16
Epoch 9 loss 1.0835307827755014 valid acc 15/16
Epoch 9 loss 1.216858290569759 valid acc 15/16
Epoch 9 loss 1.6208849385597766 valid acc 15/16
Epoch 9 loss 1.4579898282224508 valid acc 15/16
Epoch 9 loss 1.929401775440916 valid acc 15/16
Epoch 9 loss 1.0541085575949756 valid acc 15/16
Epoch 9 loss 1.2830415409982316 valid acc 15/16
Epoch 9 loss 2.5414217388388445 valid acc 15/16
Epoch 9 loss 1.7207147207191453 valid acc 15/16
Epoch 9 loss 1.8122399179487272 valid acc 15/16
Epoch 9 loss 1.1981397917679317 valid acc 15/16
Epoch 9 loss 1.4813523368012058 valid acc 15/16
Epoch 9 loss 2.255503383396702 valid acc 15/16
Epoch 10 loss 0.036776384265411154 valid acc 15/16
Epoch 10 loss 2.162951568020554 valid acc 15/16
Epoch 10 loss 1.7178250685716123 valid acc 15/16
Epoch 10 loss 1.7413446520216336 valid acc 14/16
Epoch 10 loss 0.6633442096662985 valid acc 15/16
Epoch 10 loss 1.5360783828455873 valid acc 15/16
Epoch 10 loss 1.501475758352059 valid acc 15/16
Epoch 10 loss 1.4273730118758183 valid acc 16/16
Epoch 10 loss 1.887818148369493 valid acc 15/16
Epoch 10 loss 0.8261537096438939 valid acc 15/16
Epoch 10 loss 0.8604378454608654 valid acc 15/16
Epoch 10 loss 1.7959718982470227 valid acc 16/16
Epoch 10 loss 1.7552207905712107 valid acc 16/16
Epoch 10 loss 2.2652141325372086 valid acc 15/16
Epoch 10 loss 1.9686174876404707 valid acc 16/16
Epoch 10 loss 1.518908883249921 valid acc 15/16
Epoch 10 loss 2.9036211356403605 valid acc 14/16
Epoch 10 loss 1.5451566676743174 valid acc 15/16
Epoch 10 loss 1.700775574914896 valid acc 15/16
Epoch 10 loss 2.794390379495384 valid acc 14/16
Epoch 10 loss 1.8150373171439875 valid acc 15/16
Epoch 10 loss 0.7989477701912909 valid acc 15/16
Epoch 10 loss 1.0275545630893572 valid acc 15/16
Epoch 10 loss 1.4378965623550692 valid acc 15/16
Epoch 10 loss 0.9294821416789872 valid acc 15/16
Epoch 10 loss 1.1405870291612594 valid acc 15/16
Epoch 10 loss 0.7031873820647582 valid acc 15/16
Epoch 10 loss 1.128269183407458 valid acc 15/16
Epoch 10 loss 1.672643575293427 valid acc 15/16
Epoch 10 loss 0.6359668763071613 valid acc 15/16
Epoch 10 loss 1.2239055381171684 valid acc 16/16
Epoch 10 loss 1.3532165187485723 valid acc 16/16
Epoch 10 loss 1.163345022572617 valid acc 15/16
Epoch 10 loss 0.9298434481797841 valid acc 15/16
Epoch 10 loss 1.9961837512286735 valid acc 15/16
Epoch 10 loss 1.8860495066087348 valid acc 15/16
Epoch 10 loss 1.3250894460431568 valid acc 15/16
Epoch 10 loss 0.9263860437622233 valid acc 15/16
Epoch 10 loss 0.9463180774373285 valid acc 15/16
Epoch 10 loss 1.3142149385064774 valid acc 15/16
Epoch 10 loss 1.5817109210156 valid acc 15/16
Epoch 10 loss 1.837840932973709 valid acc 15/16
Epoch 10 loss 1.2887929316176006 valid acc 15/16
Epoch 10 loss 1.3738250195933048 valid acc 15/16
Epoch 10 loss 1.7845757622918224 valid acc 15/16
Epoch 10 loss 0.9558266877673518 valid acc 15/16
Epoch 10 loss 0.9578277545331929 valid acc 15/16
Epoch 10 loss 2.233188301775084 valid acc 15/16
Epoch 10 loss 1.141349872817567 valid acc 15/16
Epoch 10 loss 1.0915845616100956 valid acc 15/16
Epoch 10 loss 1.1747381226996056 valid acc 15/16
Epoch 10 loss 1.2561182399690096 valid acc 15/16
Epoch 10 loss 1.892925014492148 valid acc 15/16
Epoch 10 loss 0.8639092313876915 valid acc 15/16
Epoch 10 loss 1.2765150585944596 valid acc 14/16
Epoch 10 loss 1.2737050121754605 valid acc 15/16
Epoch 10 loss 1.1169302908244196 valid acc 15/16
Epoch 10 loss 1.6181648522789431 valid acc 15/16
Epoch 10 loss 1.809460310716764 valid acc 14/16
Epoch 10 loss 1.8249621461346504 valid acc 15/16
Epoch 10 loss 1.078068172816569 valid acc 14/16
Epoch 10 loss 1.1705223296888645 valid acc 15/16
Epoch 10 loss 1.7609303709213546 valid acc 15/16
Epoch 11 loss 0.048970238326693484 valid acc 15/16
Epoch 11 loss 1.9858374770632974 valid acc 15/16
Epoch 11 loss 1.6164842519970857 valid acc 15/16
Epoch 11 loss 1.528328962360479 valid acc 15/16
Epoch 11 loss 0.8454629801969327 valid acc 15/16
Epoch 11 loss 0.6842150969288707 valid acc 15/16
Epoch 11 loss 1.5189073042308157 valid acc 15/16
Epoch 11 loss 1.3204970985619704 valid acc 15/16
Epoch 11 loss 1.5707403107790043 valid acc 15/16
Epoch 11 loss 1.163290937026528 valid acc 15/16
Epoch 11 loss 0.8911025130088217 valid acc 15/16
Epoch 11 loss 1.6304090117427719 valid acc 15/16
Epoch 11 loss 2.138627396285477 valid acc 16/16
Epoch 11 loss 1.9985028317209268 valid acc 15/16
Epoch 11 loss 2.079903511161634 valid acc 16/16
Epoch 11 loss 1.6084364586402642 valid acc 15/16
Epoch 11 loss 2.3940650071303913 valid acc 15/16
Epoch 11 loss 1.5510290184527888 valid acc 15/16
Epoch 11 loss 1.1253107402208058 valid acc 16/16
Epoch 11 loss 1.5892301686446761 valid acc 15/16
Epoch 11 loss 1.2071658005766126 valid acc 15/16
Epoch 11 loss 0.4803711455678642 valid acc 15/16
Epoch 11 loss 0.8456555954160622 valid acc 15/16
Epoch 11 loss 1.2214850906145063 valid acc 15/16
Epoch 11 loss 0.8354683317584022 valid acc 15/16
Epoch 11 loss 0.7974391218706598 valid acc 15/16
Epoch 11 loss 1.1403791992633598 valid acc 15/16
Epoch 11 loss 0.8844079435776453 valid acc 15/16
Epoch 11 loss 0.8382455277603366 valid acc 15/16
Epoch 11 loss 0.2079904538630783 valid acc 15/16
Epoch 11 loss 0.8178808682560047 valid acc 15/16
Epoch 11 loss 1.5465832260812935 valid acc 16/16
Epoch 11 loss 0.9344364495974942 valid acc 15/16
Epoch 11 loss 1.689395465073344 valid acc 15/16
Epoch 11 loss 2.085096170208966 valid acc 15/16
Epoch 11 loss 1.5060995831934558 valid acc 15/16
Epoch 11 loss 1.0258378328634574 valid acc 15/16
Epoch 11 loss 0.9520798400291578 valid acc 15/16
Epoch 11 loss 0.8662055761181129 valid acc 15/16
Epoch 11 loss 1.2213584234959154 valid acc 15/16
Epoch 11 loss 0.9664936560138078 valid acc 15/16
Epoch 11 loss 1.4724830809092704 valid acc 15/16
Epoch 11 loss 1.1947693290796593 valid acc 15/16
Epoch 11 loss 1.3940765363506429 valid acc 15/16
Epoch 11 loss 1.747588014668969 valid acc 15/16
Epoch 11 loss 1.419126996919368 valid acc 15/16
Epoch 11 loss 1.2350648734764755 valid acc 15/16
Epoch 11 loss 1.1573073529526685 valid acc 15/16
Epoch 11 loss 0.901693496335927 valid acc 15/16
Epoch 11 loss 0.7090970843965817 valid acc 15/16
Epoch 11 loss 0.6752440241664679 valid acc 15/16
Epoch 11 loss 0.9620530649320217 valid acc 15/16
Epoch 11 loss 1.6957851477059094 valid acc 15/16
Epoch 11 loss 1.4566570670954224 valid acc 15/16
Epoch 11 loss 1.6746212659313158 valid acc 15/16
Epoch 11 loss 1.049993058707211 valid acc 16/16
Epoch 11 loss 0.7647399546399338 valid acc 15/16
Epoch 11 loss 2.2915991957166866 valid acc 14/16
Epoch 11 loss 1.4939904969589137 valid acc 16/16
Epoch 11 loss 2.134666033808207 valid acc 14/16
Epoch 11 loss 1.5445837554893114 valid acc 14/16
Epoch 11 loss 1.1457544458305833 valid acc 15/16
Epoch 11 loss 1.5220668397575334 valid acc 14/16
Epoch 12 loss 0.02241053884553302 valid acc 14/16
Epoch 12 loss 2.006315611628623 valid acc 16/16
Epoch 12 loss 1.3540205743868365 valid acc 15/16
Epoch 12 loss 1.9381301603555663 valid acc 14/16
Epoch 12 loss 1.2183208010539237 valid acc 14/16
Epoch 12 loss 0.9953581017802975 valid acc 15/16
Epoch 12 loss 1.726959827272291 valid acc 15/16
Epoch 12 loss 1.5084264404956482 valid acc 15/16
Epoch 12 loss 1.6314709043283304 valid acc 15/16
Epoch 12 loss 0.965219078115375 valid acc 15/16
Epoch 12 loss 0.8107424035011971 valid acc 15/16
Epoch 12 loss 1.7640463235872312 valid acc 15/16
Epoch 12 loss 1.6472219604194933 valid acc 15/16
Epoch 12 loss 1.8004958416447208 valid acc 15/16
Epoch 12 loss 2.3292542213090055 valid acc 15/16
Epoch 12 loss 1.464803662101757 valid acc 15/16
Epoch 12 loss 2.0442149096980766 valid acc 15/16
Epoch 12 loss 1.3536475678358588 valid acc 15/16
Epoch 12 loss 1.4375485818940779 valid acc 14/16
Epoch 12 loss 1.152411602052653 valid acc 15/16
Epoch 12 loss 1.553399882073304 valid acc 14/16
Epoch 12 loss 0.4637823380662015 valid acc 15/16
Epoch 12 loss 0.5619188304849148 valid acc 15/16
Epoch 12 loss 0.6749248921830625 valid acc 15/16
Epoch 12 loss 0.8961804467022991 valid acc 15/16
Epoch 12 loss 0.9210655482557005 valid acc 14/16
Epoch 12 loss 0.6443584140446121 valid acc 14/16
Epoch 12 loss 1.6192441001772633 valid acc 15/16
Epoch 12 loss 0.9554760449607207 valid acc 14/16
Epoch 12 loss 0.14776720433632107 valid acc 15/16
Epoch 12 loss 1.1058043422553592 valid acc 15/16
Epoch 12 loss 1.7093872114834503 valid acc 15/16
Epoch 12 loss 0.7033725815835499 valid acc 15/16
Epoch 12 loss 0.867105980860827 valid acc 16/16
Epoch 12 loss 1.9334230403531962 valid acc 16/16
Epoch 12 loss 1.9021357058696466 valid acc 15/16
Epoch 12 loss 0.811617968248581 valid acc 15/16
Epoch 12 loss 0.8437452482252865 valid acc 15/16
Epoch 12 loss 1.0676571763820675 valid acc 16/16
Epoch 12 loss 1.1479891218615843 valid acc 16/16
Epoch 12 loss 1.4778580051568841 valid acc 15/16
Epoch 12 loss 0.945087528711865 valid acc 15/16
Epoch 12 loss 1.2268789027733322 valid acc 15/16
Epoch 12 loss 1.3236034555106153 valid acc 15/16
Epoch 12 loss 1.9175996138075084 valid acc 15/16
Epoch 12 loss 1.2206742684546044 valid acc 15/16
Epoch 12 loss 0.9107637170408078 valid acc 15/16
Epoch 12 loss 1.0501919746964865 valid acc 15/16
Epoch 12 loss 0.7959207446004176 valid acc 15/16
Epoch 12 loss 0.6442338718559887 valid acc 15/16
Epoch 12 loss 0.7018549800032472 valid acc 16/16
Epoch 12 loss 1.1305165980568992 valid acc 16/16
Epoch 12 loss 1.553503625162608 valid acc 15/16
Epoch 12 loss 1.154473682049654 valid acc 15/16
Epoch 12 loss 2.182066874411187 valid acc 15/16
Epoch 12 loss 1.196574787989812 valid acc 16/16
Epoch 12 loss 0.9389323848756457 valid acc 16/16
Epoch 12 loss 1.460884410845325 valid acc 16/16
Epoch 12 loss 1.3550859589966264 valid acc 16/16
Epoch 12 loss 1.4699732220994646 valid acc 16/16
Epoch 12 loss 1.3248504041433036 valid acc 16/16
Epoch 12 loss 0.9821923947313813 valid acc 16/16
Epoch 12 loss 1.5911061118077465 valid acc 16/16
Epoch 13 loss 0.07414005841140786 valid acc 16/16
Epoch 13 loss 1.5835801246540135 valid acc 16/16
Epoch 13 loss 1.2293594991532943 valid acc 16/16
Epoch 13 loss 1.4718765437827206 valid acc 16/16
Epoch 13 loss 0.9737637652773214 valid acc 15/16
Epoch 13 loss 1.238616729493783 valid acc 16/16
Epoch 13 loss 0.900073783407409 valid acc 16/16
Epoch 13 loss 1.8376401698850136 valid acc 16/16
Epoch 13 loss 1.4956352207088106 valid acc 16/16
Epoch 13 loss 0.9285137934027341 valid acc 16/16
Epoch 13 loss 0.8665835921893708 valid acc 15/16
Epoch 13 loss 1.5640684815254633 valid acc 15/16
Epoch 13 loss 1.6620228395294834 valid acc 15/16
Epoch 13 loss 1.1565961452638172 valid acc 15/16
Epoch 13 loss 1.6984399234839178 valid acc 16/16
Epoch 13 loss 1.3417848999826218 valid acc 16/16
Epoch 13 loss 2.190837460974726 valid acc 15/16
Epoch 13 loss 1.4720321862794779 valid acc 16/16
Epoch 13 loss 1.6578788324428508 valid acc 15/16
Epoch 13 loss 1.5954485901553546 valid acc 15/16
Epoch 13 loss 1.630512291765836 valid acc 15/16
Epoch 13 loss 0.7589363522918484 valid acc 15/16
Epoch 13 loss 0.907460332545106 valid acc 15/16
Epoch 13 loss 0.9522537223478714 valid acc 15/16
Epoch 13 loss 0.6558660090220824 valid acc 15/16
Epoch 13 loss 1.1517950451721113 valid acc 15/16
Epoch 13 loss 1.1476744693897902 valid acc 15/16
Epoch 13 loss 1.222984503379314 valid acc 15/16
Epoch 13 loss 0.9629375680982502 valid acc 15/16
Epoch 13 loss 0.35180687049642156 valid acc 15/16
Epoch 13 loss 1.049622630040023 valid acc 15/16
Epoch 13 loss 1.3266167995608538 valid acc 15/16
Epoch 13 loss 1.0637524468857602 valid acc 15/16
Epoch 13 loss 0.8463756807099526 valid acc 15/16
Epoch 13 loss 1.8346989651783434 valid acc 16/16
Epoch 13 loss 1.6308873277890694 valid acc 14/16
Epoch 13 loss 1.2538731951017281 valid acc 15/16
Epoch 13 loss 1.300699718019473 valid acc 15/16
Epoch 13 loss 1.0727235202792478 valid acc 15/16
Epoch 13 loss 1.119162687967793 valid acc 15/16
Epoch 13 loss 1.1329057565192981 valid acc 15/16
Epoch 13 loss 1.2255186402224219 valid acc 15/16
Epoch 13 loss 1.9398547083948956 valid acc 15/16
Epoch 13 loss 0.9375465276236472 valid acc 15/16
Epoch 13 loss 1.9150706241661941 valid acc 15/16
Epoch 13 loss 1.0685436911642379 valid acc 15/16
Epoch 13 loss 0.7730133581433878 valid acc 15/16
Epoch 13 loss 1.5114744278882632 valid acc 15/16
Epoch 13 loss 1.0853075150645641 valid acc 15/16
Epoch 13 loss 0.69656624549719 valid acc 15/16
Epoch 13 loss 0.9831852711126743 valid acc 15/16
Epoch 13 loss 1.0879748950768382 valid acc 15/16
Epoch 13 loss 1.529648367512676 valid acc 15/16
Epoch 13 loss 1.203127296502286 valid acc 15/16
Epoch 13 loss 1.6408110745242235 valid acc 15/16
Epoch 13 loss 0.5389652430418891 valid acc 16/16
Epoch 13 loss 0.8628691933418544 valid acc 16/16
Epoch 13 loss 1.47962268217448 valid acc 15/16
Epoch 13 loss 1.3592193103014845 valid acc 15/16
Epoch 13 loss 1.3030518491824918 valid acc 15/16
Epoch 13 loss 0.5898449202187226 valid acc 14/16
Epoch 13 loss 0.8905175951767262 valid acc 16/16
Epoch 13 loss 1.217277101884866 valid acc 15/16
Epoch 14 loss 0.1288287295585029 valid acc 15/16
Epoch 14 loss 1.668823701748169 valid acc 15/16
Epoch 14 loss 1.6292448260453694 valid acc 15/16
Epoch 14 loss 1.5524475976585257 valid acc 15/16
Epoch 14 loss 0.616084822451551 valid acc 15/16
Epoch 14 loss 0.7825400774404475 valid acc 15/16
Epoch 14 loss 2.2740507970019994 valid acc 15/16
Epoch 14 loss 1.3879349613177248 valid acc 15/16
Epoch 14 loss 1.8395604297211448 valid acc 15/16
Epoch 14 loss 0.9759513694368388 valid acc 15/16
Epoch 14 loss 1.3088009551650492 valid acc 15/16
Epoch 14 loss 1.47262709294481 valid acc 15/16
Epoch 14 loss 1.671925050434616 valid acc 15/16
Epoch 14 loss 1.4151365586434994 valid acc 15/16
Epoch 14 loss 1.4082043009649663 valid acc 15/16
Epoch 14 loss 1.4781492374950091 valid acc 15/16
Epoch 14 loss 2.103391767087742 valid acc 15/16
Epoch 14 loss 1.7183652011443518 valid acc 16/16
Epoch 14 loss 1.3083088913414687 valid acc 15/16
Epoch 14 loss 1.559818465390687 valid acc 15/16
Epoch 14 loss 0.9146595377297249 valid acc 15/16
Epoch 14 loss 0.36848159249256973 valid acc 15/16
Epoch 14 loss 0.432181338713751 valid acc 15/16
Epoch 14 loss 0.784556792759936 valid acc 15/16
Epoch 14 loss 0.653612274346986 valid acc 15/16
Epoch 14 loss 0.8796617555641367 valid acc 14/16
Epoch 14 loss 0.4859899009895131 valid acc 14/16
Epoch 14 loss 1.3494992435872566 valid acc 15/16
Epoch 14 loss 1.6342548843889184 valid acc 15/16
Epoch 14 loss 0.378807178082211 valid acc 15/16
Epoch 14 loss 1.0470806233832484 valid acc 15/16
Epoch 14 loss 1.3594202916877032 valid acc 16/16
Epoch 14 loss 0.6252681527231319 valid acc 15/16
Epoch 14 loss 1.008242303852543 valid acc 15/16
Epoch 14 loss 2.1127575971858024 valid acc 15/16
Epoch 14 loss 1.2851174609923186 valid acc 14/16
Epoch 14 loss 0.7642951458545768 valid acc 14/16
Epoch 14 loss 0.9724331932690184 valid acc 15/16
Epoch 14 loss 1.1550829022942815 valid acc 15/16
Epoch 14 loss 0.942983117567463 valid acc 15/16
Epoch 14 loss 1.0684939676969933 valid acc 15/16
Epoch 14 loss 0.8063548835102854 valid acc 15/16
Epoch 14 loss 1.0227024102177245 valid acc 15/16
Epoch 14 loss 1.378243028014821 valid acc 14/16
Epoch 14 loss 1.845163685458658 valid acc 16/16
Epoch 14 loss 0.8674781771076799 valid acc 16/16
Epoch 14 loss 1.1814024196813624 valid acc 15/16
Epoch 14 loss 1.0117446298536625 valid acc 15/16
Epoch 14 loss 0.9341597621267947 valid acc 15/16
Epoch 14 loss 0.7848129523912888 valid acc 15/16
Epoch 14 loss 0.9273044519708107 valid acc 15/16
Epoch 14 loss 0.7365180878583529 valid acc 15/16
Epoch 14 loss 1.8496954238220498 valid acc 16/16
Epoch 14 loss 1.3061451982605732 valid acc 15/16
Epoch 14 loss 1.6689551708061265 valid acc 15/16
Epoch 14 loss 0.876134691716974 valid acc 15/16
Epoch 14 loss 1.0963492904525696 valid acc 16/16
Epoch 14 loss 1.8425106037606709 valid acc 15/16
Epoch 14 loss 1.1358569405498313 valid acc 15/16
Epoch 14 loss 1.3300411539076022 valid acc 15/16
Epoch 14 loss 0.8282745150017364 valid acc 14/16
Epoch 14 loss 0.6300371202406201 valid acc 16/16
Epoch 14 loss 1.5862116236030928 valid acc 16/16
Epoch 15 loss 0.09047983699393983 valid acc 16/16
Epoch 15 loss 1.9047006021316064 valid acc 15/16
Epoch 15 loss 1.4970955159792592 valid acc 16/16
Epoch 15 loss 1.6474994709501083 valid acc 15/16
Epoch 15 loss 0.8135660631674192 valid acc 14/16
Epoch 15 loss 1.2104602385048597 valid acc 15/16
Epoch 15 loss 1.1870562712260586 valid acc 15/16
Epoch 15 loss 1.489519534799624 valid acc 15/16
Epoch 15 loss 1.6251559296471894 valid acc 15/16
Epoch 15 loss 0.7019072416076966 valid acc 15/16
Epoch 15 loss 0.8903787668814261 valid acc 15/16
Epoch 15 loss 1.784296216777707 valid acc 15/16
Epoch 15 loss 1.3543361626803712 valid acc 15/16
Epoch 15 loss 1.8244801845152743 valid acc 15/16
Epoch 15 loss 1.9110164145102704 valid acc 15/16
Epoch 15 loss 1.7599100857995544 valid acc 14/16
Epoch 15 loss 2.8918814095125795 valid acc 14/16
Epoch 15 loss 1.974866306101132 valid acc 14/16
Epoch 15 loss 1.7468983705743397 valid acc 15/16
Epoch 15 loss 1.6480636414931364 valid acc 14/16
Epoch 15 loss 1.4541688798983707 valid acc 14/16
Epoch 15 loss 0.8764178698086251 valid acc 14/16
Epoch 15 loss 0.647958517944322 valid acc 15/16
Epoch 15 loss 0.6384949231449182 valid acc 15/16
Epoch 15 loss 1.1242770230021035 valid acc 15/16
Epoch 15 loss 0.8825435207419714 valid acc 15/16
Epoch 15 loss 0.9719123068736341 valid acc 15/16
Epoch 15 loss 1.0344718988465618 valid acc 15/16
Epoch 15 loss 1.0929024508843899 valid acc 15/16
Epoch 15 loss 0.25841951458470547 valid acc 15/16
Epoch 15 loss 0.8960298551720199 valid acc 15/16
Epoch 15 loss 1.2296387589062754 valid acc 15/16
Epoch 15 loss 1.0492491484528672 valid acc 15/16
Epoch 15 loss 1.1006424927563156 valid acc 15/16
Epoch 15 loss 2.2141527447401894 valid acc 15/16
Epoch 15 loss 1.7407804162366207 valid acc 15/16
Epoch 15 loss 0.6235058592938365 valid acc 15/16
Epoch 15 loss 1.0162014147012717 valid acc 15/16
Epoch 15 loss 0.9005380977713473 valid acc 15/16
Epoch 15 loss 0.9911153442960784 valid acc 15/16
Epoch 15 loss 0.8798899000574667 valid acc 15/16
Epoch 15 loss 0.8571681965992807 valid acc 15/16
Epoch 15 loss 1.6941376029767907 valid acc 15/16
Epoch 15 loss 0.9162663839904985 valid acc 15/16
Epoch 15 loss 1.6249482742889898 valid acc 15/16
Epoch 15 loss 0.8754637614151293 valid acc 15/16
Epoch 15 loss 1.0759506862687964 valid acc 15/16
Epoch 15 loss 1.2682249204980711 valid acc 15/16
Epoch 15 loss 0.9847662505177939 valid acc 15/16
Epoch 15 loss 0.5829407643101288 valid acc 15/16
Epoch 15 loss 1.0378877910497233 valid acc 16/16
Epoch 15 loss 1.255425564761132 valid acc 15/16
Epoch 15 loss 1.574703084737692 valid acc 15/16
Epoch 15 loss 0.8770405210739589 valid acc 15/16
Epoch 15 loss 1.5538945541573208 valid acc 15/16
Epoch 15 loss 0.941233629776107 valid acc 15/16
Epoch 15 loss 0.5415254864698035 valid acc 16/16
Epoch 15 loss 1.7008639008248392 valid acc 15/16
Epoch 15 loss 0.6650499839390672 valid acc 15/16
Epoch 15 loss 1.349121163982964 valid acc 15/16
Epoch 15 loss 1.314725371501784 valid acc 15/16
Epoch 15 loss 0.9101555037333183 valid acc 16/16
Epoch 15 loss 1.5728658368113302 valid acc 16/16
Epoch 16 loss 0.11663147703802977 valid acc 16/16
Epoch 16 loss 1.6740937188038225 valid acc 16/16
Epoch 16 loss 1.4250424530786727 valid acc 16/16
Epoch 16 loss 1.549544829228697 valid acc 16/16
Epoch 16 loss 0.5986615817305253 valid acc 15/16
Epoch 16 loss 0.8264474625028937 valid acc 16/16
Epoch 16 loss 2.08928664796513 valid acc 16/16
Epoch 16 loss 1.519169852389563 valid acc 14/16
Epoch 16 loss 1.8310006847494087 valid acc 16/16
Epoch 16 loss 0.9932900920998661 valid acc 15/16
Epoch 16 loss 1.0065632080498406 valid acc 15/16
Epoch 16 loss 1.934818580039134 valid acc 15/16
Epoch 16 loss 1.4661471420844645 valid acc 15/16
Epoch 16 loss 1.9504094360727684 valid acc 15/16
Epoch 16 loss 1.8581054543012545 valid acc 16/16
Epoch 16 loss 0.9575023942221761 valid acc 16/16
Epoch 16 loss 3.0230318114215393 valid acc 16/16
Epoch 16 loss 2.194325882471997 valid acc 16/16
Epoch 16 loss 1.2596679904801351 valid acc 16/16
Epoch 16 loss 1.323570662319569 valid acc 16/16
Epoch 16 loss 1.2706846388353985 valid acc 15/16
Epoch 16 loss 0.5973022906053649 valid acc 15/16
Epoch 16 loss 0.751364669783413 valid acc 15/16
Epoch 16 loss 0.7120117403328545 valid acc 15/16
Epoch 16 loss 0.9809114978724343 valid acc 15/16
Epoch 16 loss 0.9913825158203847 valid acc 15/16
Epoch 16 loss 1.0240152245334968 valid acc 15/16
Epoch 16 loss 1.0954677363747793 valid acc 14/16
Epoch 16 loss 1.011418888694437 valid acc 14/16
Epoch 16 loss 0.9915108622946757 valid acc 15/16
Epoch 16 loss 0.8581798655590624 valid acc 15/16
Epoch 16 loss 1.514912863364751 valid acc 15/16
Epoch 16 loss 1.2046912402622465 valid acc 15/16
Epoch 16 loss 1.0503125063082277 valid acc 15/16
Epoch 16 loss 2.0006908541462276 valid acc 15/16
Epoch 16 loss 0.7368127978731909 valid acc 15/16
Epoch 16 loss 0.860982532061942 valid acc 15/16
Epoch 16 loss 1.2776403447399822 valid acc 15/16
Epoch 16 loss 1.0345954047985753 valid acc 15/16
Epoch 16 loss 1.0876091400681682 valid acc 15/16
Epoch 16 loss 0.8358307282375114 valid acc 15/16
Epoch 16 loss 0.8786539726132425 valid acc 15/16
Epoch 16 loss 1.5384800635552354 valid acc 15/16
Epoch 16 loss 0.902074723483419 valid acc 15/16
Epoch 16 loss 1.7641461174558197 valid acc 15/16
Epoch 16 loss 1.1274638898033351 valid acc 15/16
Epoch 16 loss 1.263702522377844 valid acc 15/16
Epoch 16 loss 1.2388348126231903 valid acc 15/16
Epoch 16 loss 1.0206715939538251 valid acc 15/16
Epoch 16 loss 0.8148601120341015 valid acc 15/16
Epoch 16 loss 0.8232380022769252 valid acc 15/16
Epoch 16 loss 1.2811558709699993 valid acc 15/16
Epoch 16 loss 1.78068624736573 valid acc 15/16
Epoch 16 loss 1.2786085819202933 valid acc 15/16
Epoch 16 loss 1.293948008046903 valid acc 15/16
Epoch 16 loss 0.5153292659761908 valid acc 15/16
Epoch 16 loss 1.0484161768120053 valid acc 15/16
Epoch 16 loss 1.281458135730775 valid acc 16/16
Epoch 16 loss 0.9782050579175869 valid acc 15/16
Epoch 16 loss 1.3249212445610694 valid acc 15/16
Epoch 16 loss 1.1570542057906865 valid acc 15/16
Epoch 16 loss 1.2583055090250657 valid acc 16/16
Epoch 16 loss 1.290426703223748 valid acc 16/16
Epoch 17 loss 0.15258163138724 valid acc 16/16
Epoch 17 loss 1.6583429611505818 valid acc 16/16
Epoch 17 loss 1.0032659276633527 valid acc 15/16
Epoch 17 loss 1.328437226078058 valid acc 15/16
Epoch 17 loss 1.1165116642156194 valid acc 15/16
Epoch 17 loss 0.7346800626741119 valid acc 15/16
Epoch 17 loss 1.676406942931739 valid acc 16/16
Epoch 17 loss 1.6191453808962244 valid acc 15/16
Epoch 17 loss 1.8869439052987893 valid acc 15/16
Epoch 17 loss 0.8309025656175377 valid acc 15/16
Epoch 17 loss 0.8832973126542828 valid acc 15/16
Epoch 17 loss 1.3817074696114857 valid acc 15/16
Epoch 17 loss 1.2968549916037289 valid acc 15/16
Epoch 17 loss 1.5911430143135983 valid acc 15/16
Epoch 17 loss 1.9496978971483312 valid acc 16/16
Epoch 17 loss 2.0113223907486444 valid acc 16/16
Epoch 17 loss 2.5771574304131812 valid acc 16/16
Epoch 17 loss 1.4749900099803102 valid acc 15/16
Epoch 17 loss 1.462898683784651 valid acc 15/16
Epoch 17 loss 1.5184491940519762 valid acc 15/16
Epoch 17 loss 1.3134096826327708 valid acc 15/16
Epoch 17 loss 0.7405655675312266 valid acc 15/16
Epoch 17 loss 0.5444437048598753 valid acc 15/16
Epoch 17 loss 0.6565080070982773 valid acc 15/16
Epoch 17 loss 1.1796611456121293 valid acc 15/16
Epoch 17 loss 1.2074837681115358 valid acc 15/16
Epoch 17 loss 0.8517710731420964 valid acc 15/16
Epoch 17 loss 0.7249725977579411 valid acc 15/16
Epoch 17 loss 0.6451234738141228 valid acc 15/16
Epoch 17 loss 0.5006801792764918 valid acc 15/16
Epoch 17 loss 0.9887464603877072 valid acc 16/16
Epoch 17 loss 1.260698556483248 valid acc 16/16
Epoch 17 loss 0.8142433918508879 valid acc 15/16
Epoch 17 loss 0.6302171851758874 valid acc 15/16
Epoch 17 loss 2.1808960018736565 valid acc 16/16
Epoch 17 loss 1.3043097615858363 valid acc 15/16
Epoch 17 loss 0.7612713130587233 valid acc 15/16
Epoch 17 loss 1.090281479157451 valid acc 15/16
Epoch 17 loss 1.1323476363966405 valid acc 15/16
Epoch 17 loss 1.4050873810782538 valid acc 15/16
Epoch 17 loss 1.0408374652755876 valid acc 15/16
Epoch 17 loss 0.5691159609329556 valid acc 15/16
Epoch 17 loss 1.3484786332930478 valid acc 15/16
Epoch 17 loss 0.7490866035443451 valid acc 14/16
Epoch 17 loss 1.0256253474832262 valid acc 16/16
Epoch 17 loss 0.8134441576246765 valid acc 15/16
Epoch 17 loss 1.254145264064608 valid acc 15/16
Epoch 17 loss 1.2819735340104517 valid acc 15/16
Epoch 17 loss 1.0216447030536269 valid acc 15/16
Epoch 17 loss 0.57903584121443 valid acc 15/16
Epoch 17 loss 0.6873873953146603 valid acc 15/16
Epoch 17 loss 1.7224763620116743 valid acc 16/16
Epoch 17 loss 1.5746878620620324 valid acc 15/16
Epoch 17 loss 1.3711737133480846 valid acc 15/16
Epoch 17 loss 1.5242234322717767 valid acc 15/16
Epoch 17 loss 0.6870610081704704 valid acc 15/16
Epoch 17 loss 1.3140191003481512 valid acc 15/16
Epoch 17 loss 1.1547393107757002 valid acc 16/16
Epoch 17 loss 1.0975772049977786 valid acc 14/16
Epoch 17 loss 0.675878484442948 valid acc 15/16
Epoch 17 loss 0.7511383784366702 valid acc 16/16
Epoch 17 loss 0.5846450322272737 valid acc 16/16
Epoch 17 loss 2.1160642426352974 valid acc 16/16
Epoch 18 loss 0.07684200388410006 valid acc 15/16
Epoch 18 loss 1.8709380944177414 valid acc 16/16
Epoch 18 loss 1.0650572608186442 valid acc 16/16
Epoch 18 loss 1.0817079389993294 valid acc 16/16
Epoch 18 loss 0.8967009816572079 valid acc 15/16
Epoch 18 loss 0.6990443294323991 valid acc 16/16
Epoch 18 loss 1.593187594309566 valid acc 16/16
Epoch 18 loss 1.071018300395768 valid acc 15/16
Epoch 18 loss 1.531641342023608 valid acc 15/16
Epoch 18 loss 0.9300486606014926 valid acc 15/16
Epoch 18 loss 1.2907171182239803 valid acc 15/16
Epoch 18 loss 1.5467424201397595 valid acc 15/16
Epoch 18 loss 1.6778307447922942 valid acc 15/16
Epoch 18 loss 1.64996768340883 valid acc 14/16
Epoch 18 loss 1.3432366503674213 valid acc 16/16
Epoch 18 loss 0.8376200666247413 valid acc 15/16
Epoch 18 loss 1.7295434600895365 valid acc 14/16
Epoch 18 loss 1.2314958643150782 valid acc 14/16
Epoch 18 loss 1.2896217862468378 valid acc 15/16
Epoch 18 loss 1.276663798872845 valid acc 14/16
Epoch 18 loss 1.4155204536371402 valid acc 15/16
Epoch 18 loss 0.6706744390233259 valid acc 15/16
Epoch 18 loss 0.6174182340625086 valid acc 15/16
Epoch 18 loss 1.0790115433033307 valid acc 15/16
Epoch 18 loss 1.2628700963810873 valid acc 15/16
Epoch 18 loss 0.9719905169365806 valid acc 15/16
Epoch 18 loss 1.110282947028395 valid acc 15/16
Epoch 18 loss 0.6921311927467895 valid acc 15/16
Epoch 18 loss 0.7678482368193236 valid acc 15/16
Epoch 18 loss 0.5463374600057174 valid acc 15/16
Epoch 18 loss 0.8802824955953448 valid acc 16/16
Epoch 18 loss 1.052110926160965 valid acc 16/16
Epoch 18 loss 1.1553879922178607 valid acc 16/16
Epoch 18 loss 1.127218129466922 valid acc 16/16
Epoch 18 loss 2.0953024462315066 valid acc 16/16
Epoch 18 loss 1.0142612487599356 valid acc 14/16
Epoch 18 loss 1.1920970945783103 valid acc 14/16
Epoch 18 loss 1.1133464756561393 valid acc 14/16
Epoch 18 loss 0.7452386013597463 valid acc 15/16
Epoch 18 loss 1.1018672383484187 valid acc 16/16
Epoch 18 loss 1.0139420783702344 valid acc 15/16
Epoch 18 loss 0.6822836071246078 valid acc 16/16
Epoch 18 loss 0.7887039878694847 valid acc 14/16
Epoch 18 loss 0.7328155310882343 valid acc 14/16
Epoch 18 loss 1.2752684283789866 valid acc 15/16
Epoch 18 loss 0.8421544795798808 valid acc 16/16
Epoch 18 loss 0.7806210230570538 valid acc 15/16
Epoch 18 loss 1.296860114349953 valid acc 15/16
Epoch 18 loss 0.9677537547075605 valid acc 15/16
Epoch 18 loss 0.7616210344490464 valid acc 15/16
Epoch 18 loss 1.1616107488613647 valid acc 15/16
Epoch 18 loss 1.3593867305860299 valid acc 15/16
Epoch 18 loss 1.8436118991973387 valid acc 15/16
Epoch 18 loss 0.7652787820507121 valid acc 15/16
Epoch 18 loss 1.630182904878097 valid acc 15/16
Epoch 18 loss 0.8899667775676714 valid acc 15/16
Epoch 18 loss 1.1345872519814924 valid acc 15/16
Epoch 18 loss 1.4347390537583946 valid acc 16/16
Epoch 18 loss 0.7058367573198678 valid acc 15/16
Epoch 18 loss 1.236594819429398 valid acc 15/16
Epoch 18 loss 0.8318403842902834 valid acc 14/16
Epoch 18 loss 0.7756676599589738 valid acc 15/16
Epoch 18 loss 1.4454604823080217 valid acc 16/16
Epoch 19 loss 0.1668766109405159 valid acc 15/16
Epoch 19 loss 2.281531962130281 valid acc 15/16
Epoch 19 loss 1.2373556205818355 valid acc 15/16
Epoch 19 loss 1.6387603573741178 valid acc 15/16
Epoch 19 loss 0.3984280050219364 valid acc 14/16
Epoch 19 loss 0.8385175321016178 valid acc 16/16
Epoch 19 loss 1.1919531255954354 valid acc 16/16
Epoch 19 loss 1.601672142355677 valid acc 14/16
Epoch 19 loss 1.8352724337454136 valid acc 15/16
Epoch 19 loss 0.9697584225056445 valid acc 15/16
Epoch 19 loss 1.4329891571255806 valid acc 15/16
Epoch 19 loss 2.10268255384623 valid acc 15/16
Epoch 19 loss 1.89935190563307 valid acc 15/16
Epoch 19 loss 1.9151606612541214 valid acc 15/16
Epoch 19 loss 1.9052825268653173 valid acc 15/16
Epoch 19 loss 1.2771382963415003 valid acc 15/16
Epoch 19 loss 2.228730506929307 valid acc 15/16
Epoch 19 loss 1.7920298911703942 valid acc 16/16
Epoch 19 loss 1.549018731066572 valid acc 15/16
Epoch 19 loss 1.085968951100364 valid acc 14/16
Epoch 19 loss 1.2040981884248538 valid acc 14/16
Epoch 19 loss 0.4160579975522586 valid acc 14/16
Epoch 19 loss 0.7041608231132734 valid acc 14/16
Epoch 19 loss 0.38337978387605953 valid acc 14/16
Epoch 19 loss 1.3822746869938478 valid acc 14/16
Epoch 19 loss 1.5127094042510851 valid acc 15/16
Epoch 19 loss 0.6900267191459072 valid acc 15/16
Epoch 19 loss 0.7846814623674161 valid acc 14/16
Epoch 19 loss 1.0636073919648392 valid acc 15/16
Epoch 19 loss 0.17332729470931021 valid acc 15/16
Epoch 19 loss 0.8135805441110063 valid acc 16/16
Epoch 19 loss 1.0002159620318625 valid acc 16/16
Epoch 19 loss 0.8316923285528971 valid acc 15/16
Epoch 19 loss 0.7426981522557259 valid acc 15/16
Epoch 19 loss 2.1747964712068186 valid acc 16/16
Epoch 19 loss 1.4048694078034771 valid acc 15/16
Epoch 19 loss 1.0207423812314076 valid acc 15/16
Epoch 19 loss 1.5238341530707578 valid acc 15/16
Epoch 19 loss 0.8909626222934747 valid acc 16/16
Epoch 19 loss 1.2993276869283796 valid acc 16/16
Epoch 19 loss 0.8769690718520665 valid acc 15/16
Epoch 19 loss 1.295485886274918 valid acc 15/16
Epoch 19 loss 0.8793361986541146 valid acc 15/16
Epoch 19 loss 1.265406456554469 valid acc 16/16
Epoch 19 loss 1.190252534744523 valid acc 16/16
Epoch 19 loss 0.9862976699568582 valid acc 15/16
Epoch 19 loss 0.7534294248553515 valid acc 15/16
Epoch 19 loss 1.0415192140262541 valid acc 15/16
Epoch 19 loss 1.2375663004887283 valid acc 15/16
Epoch 19 loss 0.5064879174236211 valid acc 15/16
Epoch 19 loss 0.9255206883127072 valid acc 15/16
Epoch 19 loss 1.5887306897004068 valid acc 15/16
Epoch 19 loss 0.9898988652889321 valid acc 15/16
Epoch 19 loss 1.1547347582402803 valid acc 15/16
Epoch 19 loss 1.9899885750243043 valid acc 15/16
Epoch 19 loss 0.531701752639517 valid acc 15/16
Epoch 19 loss 1.1283015984243407 valid acc 15/16
Epoch 19 loss 1.4361266080724664 valid acc 14/16
Epoch 19 loss 1.0668168516633219 valid acc 15/16
Epoch 19 loss 0.8650642389199751 valid acc 15/16
Epoch 19 loss 1.024412958086075 valid acc 15/16
Epoch 19 loss 0.7701186996485807 valid acc 15/16
Epoch 19 loss 1.246084812695986 valid acc 15/16
Epoch 20 loss 0.03755070549942579 valid acc 15/16
Epoch 20 loss 1.5293513109098138 valid acc 15/16
Epoch 20 loss 1.3283528837891192 valid acc 15/16
Epoch 20 loss 1.1611106250945207 valid acc 16/16
Epoch 20 loss 0.9420266705277723 valid acc 15/16
Epoch 20 loss 0.7764515012458506 valid acc 15/16
Epoch 20 loss 1.6197809229073747 valid acc 15/16
Epoch 20 loss 1.5785577002450815 valid acc 15/16
Epoch 20 loss 1.6624512802298645 valid acc 15/16
Epoch 20 loss 0.756934764433313 valid acc 15/16
Epoch 20 loss 0.5172540971695444 valid acc 15/16
Epoch 20 loss 1.5225928241315554 valid acc 15/16
Epoch 20 loss 1.6774072667373714 valid acc 16/16
Epoch 20 loss 2.4076372103188124 valid acc 15/16
Epoch 20 loss 1.8128318994263153 valid acc 16/16
Epoch 20 loss 0.9284358284401086 valid acc 15/16
Epoch 20 loss 2.2445053952215046 valid acc 15/16
Epoch 20 loss 1.4084541364380856 valid acc 15/16
Epoch 20 loss 1.2985336456508205 valid acc 15/16
Epoch 20 loss 1.4220313472961748 valid acc 14/16
Epoch 20 loss 1.23152281294985 valid acc 15/16
Epoch 20 loss 0.5462744975659242 valid acc 15/16
Epoch 20 loss 0.5404062637323208 valid acc 15/16
Epoch 20 loss 0.6307390206763206 valid acc 15/16
Epoch 20 loss 0.9316150734461607 valid acc 15/16
Epoch 20 loss 0.7109296704671425 valid acc 15/16
Epoch 20 loss 0.6230589187323973 valid acc 15/16
Epoch 20 loss 0.7279771997083989 valid acc 15/16
Epoch 20 loss 1.0090203183441009 valid acc 15/16
Epoch 20 loss 0.3270322643689825 valid acc 15/16
Epoch 20 loss 1.0312870377383228 valid acc 16/16
Epoch 20 loss 1.0459011767457371 valid acc 16/16
Epoch 20 loss 1.0078025021859276 valid acc 15/16
Epoch 20 loss 1.0929481377213683 valid acc 15/16
Epoch 20 loss 1.894938112181153 valid acc 16/16
Epoch 20 loss 1.2974612857275114 valid acc 15/16
Epoch 20 loss 1.4900317875550313 valid acc 15/16
Epoch 20 loss 0.8591023965308229 valid acc 15/16
Epoch 20 loss 1.2274696287795077 valid acc 15/16
Epoch 20 loss 0.9338097745801721 valid acc 15/16
Epoch 20 loss 0.5903467924191725 valid acc 15/16
Epoch 20 loss 0.7005589121534659 valid acc 15/16
Epoch 20 loss 1.215766870346389 valid acc 15/16
Epoch 20 loss 0.9285751175378597 valid acc 15/16
Epoch 20 loss 1.5834760477991507 valid acc 15/16
Epoch 20 loss 0.9844214146191237 valid acc 15/16
Epoch 20 loss 0.8187511658023564 valid acc 15/16
Epoch 20 loss 1.0332554381696226 valid acc 15/16
Epoch 20 loss 1.0699911430469937 valid acc 15/16
Epoch 20 loss 0.36918913248502905 valid acc 15/16
Epoch 20 loss 0.5728669569220481 valid acc 15/16
Epoch 20 loss 1.1071660615066352 valid acc 15/16
Epoch 20 loss 1.3117888216837204 valid acc 15/16
Epoch 20 loss 1.1962886041984069 valid acc 15/16
Epoch 20 loss 1.4444824756502967 valid acc 15/16
Epoch 20 loss 0.5913841110367346 valid acc 15/16
Epoch 20 loss 1.0810338645777693 valid acc 15/16
Epoch 20 loss 1.3718724977677366 valid acc 15/16
Epoch 20 loss 1.3106808283677422 valid acc 15/16
Epoch 20 loss 1.1005545228846632 valid acc 15/16
Epoch 20 loss 1.3240122696323695 valid acc 14/16
Epoch 20 loss 0.7022826150591244 valid acc 15/16
Epoch 20 loss 1.1963342032607032 valid acc 16/16
Epoch 21 loss 0.15109085646218232 valid acc 15/16
Epoch 21 loss 2.081376617320652 valid acc 16/16
Epoch 21 loss 1.463521081766359 valid acc 15/16
Epoch 21 loss 1.6034397758118915 valid acc 15/16
Epoch 21 loss 0.5219973388705121 valid acc 15/16
Epoch 21 loss 1.1979958639807171 valid acc 16/16
Epoch 21 loss 1.8178941489481 valid acc 16/16
Epoch 21 loss 1.4171965619734512 valid acc 16/16
Epoch 21 loss 1.2189787110855628 valid acc 15/16
Epoch 21 loss 0.6140756249786974 valid acc 15/16
Epoch 21 loss 1.0570862042256923 valid acc 15/16
Epoch 21 loss 2.0582330797059143 valid acc 15/16
Epoch 21 loss 1.2236361477647104 valid acc 15/16
Epoch 21 loss 1.9532793546316867 valid acc 15/16
Epoch 21 loss 2.114711784172465 valid acc 16/16
Epoch 21 loss 1.0395644610266532 valid acc 16/16
Epoch 21 loss 2.189551056845189 valid acc 16/16
Epoch 21 loss 1.222118397721181 valid acc 16/16
Epoch 21 loss 1.2307368221135957 valid acc 15/16
Epoch 21 loss 1.3002276149331125 valid acc 15/16
Epoch 21 loss 1.0963018776073834 valid acc 15/16
Epoch 21 loss 0.5923937881067097 valid acc 15/16
Epoch 21 loss 0.5243968376241719 valid acc 15/16
Epoch 21 loss 0.5491127746437676 valid acc 15/16
Epoch 21 loss 1.067954187831948 valid acc 15/16
Epoch 21 loss 0.9662449882676635 valid acc 14/16
Epoch 21 loss 0.4320356033652259 valid acc 15/16
Epoch 21 loss 0.6095382816554277 valid acc 15/16
Epoch 21 loss 0.8178007761405479 valid acc 15/16
Epoch 21 loss 0.3454321945521934 valid acc 15/16
Epoch 21 loss 0.622303621874204 valid acc 16/16
Epoch 21 loss 0.926938654419392 valid acc 16/16
Epoch 21 loss 0.543558698913998 valid acc 15/16
Epoch 21 loss 0.8283537225155168 valid acc 15/16
Epoch 21 loss 1.9573942176578836 valid acc 15/16
Epoch 21 loss 1.5467448249481828 valid acc 15/16
Epoch 21 loss 0.45622120968935775 valid acc 15/16
Epoch 21 loss 0.7103566559115433 valid acc 15/16
Epoch 21 loss 0.702741119084143 valid acc 16/16
Epoch 21 loss 0.8348746552405907 valid acc 16/16
Epoch 21 loss 0.8777295167187147 valid acc 16/16
Epoch 21 loss 0.8932852841497371 valid acc 16/16
Epoch 21 loss 0.9705575441539598 valid acc 15/16
Epoch 21 loss 0.6355519336486981 valid acc 16/16
Epoch 21 loss 1.19420724161097 valid acc 16/16
Epoch 21 loss 0.6730872092483562 valid acc 16/16
Epoch 21 loss 1.2243299272471042 valid acc 16/16
Epoch 21 loss 1.1948194347035794 valid acc 16/16
Epoch 21 loss 0.7074623248675227 valid acc 15/16
Epoch 21 loss 0.742699612561185 valid acc 15/16
Epoch 21 loss 0.4594466953255005 valid acc 15/16
Epoch 21 loss 1.0443042350095024 valid acc 16/16
Epoch 21 loss 1.5051645528006423 valid acc 15/16
Epoch 21 loss 0.5811777851242902 valid acc 16/16
Epoch 21 loss 1.1270897694309516 valid acc 15/16
Epoch 21 loss 0.5843902681837596 valid acc 16/16
Epoch 21 loss 1.064909223109578 valid acc 15/16
Epoch 21 loss 1.1482771590299308 valid acc 15/16
Epoch 21 loss 0.9722643157296915 valid acc 15/16
Epoch 21 loss 0.9359732408173275 valid acc 16/16
Epoch 21 loss 0.9305962088970778 valid acc 15/16
Epoch 21 loss 0.8898589347388012 valid acc 16/16
Epoch 21 loss 1.1405160343836518 valid acc 16/16
Epoch 22 loss 0.10317010937844331 valid acc 16/16
Epoch 22 loss 1.2741026094376113 valid acc 16/16
Epoch 22 loss 1.4585754251062601 valid acc 16/16
Epoch 22 loss 1.0950198755139582 valid acc 16/16
Epoch 22 loss 0.8507734857572758 valid acc 15/16
Epoch 22 loss 1.0513341558261313 valid acc 16/16
Epoch 22 loss 1.7958828276815502 valid acc 16/16
Epoch 22 loss 0.9655844026305739 valid acc 16/16
Epoch 22 loss 1.5171208171359476 valid acc 15/16
Epoch 22 loss 0.7266516076574819 valid acc 15/16
Epoch 22 loss 1.1745656189812959 valid acc 15/16
Epoch 22 loss 0.9185773492351609 valid acc 16/16
Epoch 22 loss 1.3411018427108166 valid acc 16/16
Epoch 22 loss 1.9795670601560382 valid acc 15/16
Epoch 22 loss 1.6644791305541564 valid acc 16/16
Epoch 22 loss 1.2669140697382106 valid acc 16/16
Epoch 22 loss 2.789133627584429 valid acc 16/16
Epoch 22 loss 1.357942186525637 valid acc 16/16
Epoch 22 loss 1.5338022829706464 valid acc 15/16
Epoch 22 loss 1.1066224254338715 valid acc 15/16
Epoch 22 loss 1.1376527974308863 valid acc 16/16
Epoch 22 loss 0.7779240036899392 valid acc 16/16
Epoch 22 loss 0.6958306477622132 valid acc 16/16
Epoch 22 loss 0.3291084206785631 valid acc 16/16
Epoch 22 loss 1.016219297917579 valid acc 15/16
Epoch 22 loss 1.2593493283086126 valid acc 14/16
Epoch 22 loss 0.967424793310883 valid acc 16/16
Epoch 22 loss 0.7696399368271832 valid acc 15/16
Epoch 22 loss 0.9758400068771907 valid acc 15/16
Epoch 22 loss 0.27036459801991586 valid acc 15/16
Epoch 22 loss 0.5362340690677779 valid acc 16/16
Epoch 22 loss 1.1623578830392576 valid acc 16/16
Epoch 22 loss 0.759633156476174 valid acc 15/16
Epoch 22 loss 0.9117623023541508 valid acc 15/16
Epoch 22 loss 2.412999341642463 valid acc 16/16
Epoch 22 loss 1.0981290566344275 valid acc 14/16
Epoch 22 loss 0.996677797034678 valid acc 16/16
Epoch 22 loss 0.9730782557218233 valid acc 15/16
Epoch 22 loss 0.9014104457788055 valid acc 16/16
Epoch 22 loss 1.2331879454163608 valid acc 16/16
Epoch 22 loss 1.2819721496072818 valid acc 16/16
Epoch 22 loss 0.7160717145738177 valid acc 16/16
Epoch 22 loss 1.1559063653797865 valid acc 15/16
Epoch 22 loss 0.4183098869003753 valid acc 15/16
Epoch 22 loss 1.3432268811682597 valid acc 16/16
Epoch 22 loss 1.2321193690289347 valid acc 16/16
Epoch 22 loss 1.3134108500186592 valid acc 15/16
Epoch 22 loss 1.3365557766029819 valid acc 15/16
Epoch 22 loss 1.2779424713922207 valid acc 16/16
Epoch 22 loss 0.811397359642783 valid acc 15/16
Epoch 22 loss 0.673163856845006 valid acc 16/16
Epoch 22 loss 1.2384902718929227 valid acc 16/16
Epoch 22 loss 1.1783365517445352 valid acc 16/16
Epoch 22 loss 0.8341803462290698 valid acc 16/16
Epoch 22 loss 1.563431141291757 valid acc 16/16
Epoch 22 loss 0.4519855439588418 valid acc 16/16
Epoch 22 loss 0.7639701607506715 valid acc 16/16
Epoch 22 loss 1.194505705756101 valid acc 16/16
Epoch 22 loss 0.8634707667235006 valid acc 15/16
Epoch 22 loss 0.7292304265269036 valid acc 16/16
Epoch 22 loss 0.8778980341319029 valid acc 15/16
Epoch 22 loss 0.6504324074350185 valid acc 16/16
Epoch 22 loss 0.9835595875624623 valid acc 16/16
Epoch 23 loss 0.038360664525638674 valid acc 16/16
Epoch 23 loss 1.9666182032208122 valid acc 16/16
Epoch 23 loss 1.3170870285749956 valid acc 16/16
Epoch 23 loss 1.1278669135728203 valid acc 16/16
Epoch 23 loss 0.8668798543432121 valid acc 15/16
Epoch 23 loss 0.7116386543143722 valid acc 16/16
Epoch 23 loss 2.401214619773511 valid acc 16/16
Epoch 23 loss 1.5487611977810676 valid acc 16/16
Epoch 23 loss 1.750973299127806 valid acc 16/16
Epoch 23 loss 0.7036623993305069 valid acc 16/16
Epoch 23 loss 0.9836551817268235 valid acc 16/16
Epoch 23 loss 2.123797453424225 valid acc 15/16
Epoch 23 loss 1.4481647972497598 valid acc 15/16
Epoch 23 loss 1.6276311490557693 valid acc 15/16
Epoch 23 loss 1.4935361358138435 valid acc 16/16
Epoch 23 loss 0.9080465587969013 valid acc 15/16
Epoch 23 loss 1.5905230869183773 valid acc 15/16
Epoch 23 loss 1.3352597478459987 valid acc 15/16
Epoch 23 loss 1.389625103886356 valid acc 15/16
Epoch 23 loss 1.4323994980084735 valid acc 15/16
Epoch 23 loss 1.1932615213606488 valid acc 14/16
Epoch 23 loss 0.43978561740428335 valid acc 15/16
Epoch 23 loss 0.35086464673055073 valid acc 14/16
Epoch 23 loss 0.4635567447028981 valid acc 14/16
Epoch 23 loss 0.7503735341289719 valid acc 14/16
Epoch 23 loss 1.0198451930423391 valid acc 14/16
Epoch 23 loss 0.5453531839597308 valid acc 14/16
Epoch 23 loss 0.8099038435931581 valid acc 14/16
Epoch 23 loss 1.4786410863435226 valid acc 14/16
Epoch 23 loss 0.21575676682805572 valid acc 15/16
Epoch 23 loss 0.7955190166126176 valid acc 16/16
Epoch 23 loss 1.0005128561546361 valid acc 16/16
Epoch 23 loss 0.670185072317449 valid acc 15/16
Epoch 23 loss 1.4502382242119753 valid acc 16/16
Epoch 23 loss 2.1444036698250373 valid acc 16/16
Epoch 23 loss 0.9518693949874926 valid acc 15/16
Epoch 23 loss 0.6464809246293145 valid acc 16/16
Epoch 23 loss 1.059370901593103 valid acc 16/16
Epoch 23 loss 0.8416799468665764 valid acc 16/16
Epoch 23 loss 0.9700909104855747 valid acc 16/16
Epoch 23 loss 0.8173576030219201 valid acc 16/16
Epoch 23 loss 0.7152267638263591 valid acc 16/16
Epoch 23 loss 1.2415208196593317 valid acc 15/16
Epoch 23 loss 0.6881304583097928 valid acc 15/16
Epoch 23 loss 1.4303505964179695 valid acc 16/16
Epoch 23 loss 0.6376090192935775 valid acc 16/16
Epoch 23 loss 1.305951248999329 valid acc 16/16
Epoch 23 loss 1.3113330548875357 valid acc 16/16
Epoch 23 loss 1.0205940727754093 valid acc 16/16
Epoch 23 loss 0.8813534806411221 valid acc 15/16
Epoch 23 loss 0.7984044676331613 valid acc 15/16
Epoch 23 loss 0.909941589536794 valid acc 16/16
Epoch 23 loss 1.5256416498776793 valid acc 15/16
Epoch 23 loss 0.8768061615538215 valid acc 15/16
Epoch 23 loss 1.2487933537242966 valid acc 14/16
Epoch 23 loss 0.8502156663285977 valid acc 15/16
Epoch 23 loss 0.8701122989313648 valid acc 16/16
Epoch 23 loss 0.7850510014333346 valid acc 15/16
Epoch 23 loss 1.2457366732488195 valid acc 15/16
Epoch 23 loss 0.7654505349558591 valid acc 16/16
Epoch 23 loss 1.2169259228568636 valid acc 15/16
Epoch 23 loss 0.9808984474310889 valid acc 16/16
Epoch 23 loss 1.2150544278890523 valid acc 16/16
Epoch 24 loss 0.14485368542698052 valid acc 16/16
Epoch 24 loss 1.7328192862720702 valid acc 16/16
Epoch 24 loss 1.6566960973038047 valid acc 16/16
Epoch 24 loss 1.0639396586366467 valid acc 16/16
Epoch 24 loss 1.0610756403543857 valid acc 16/16
Epoch 24 loss 1.0548495777346938 valid acc 16/16
Epoch 24 loss 1.5287812420929634 valid acc 16/16
Epoch 24 loss 1.4390218679004587 valid acc 16/16
Epoch 24 loss 0.9597102110784169 valid acc 16/16
Epoch 24 loss 0.768297827103063 valid acc 16/16
Epoch 24 loss 0.8445740316096799 valid acc 16/16
Epoch 24 loss 1.6257884518495451 valid acc 16/16
Epoch 24 loss 1.3128926292139473 valid acc 16/16
Epoch 24 loss 1.3701187133532127 valid acc 16/16
Epoch 24 loss 1.4710715752602452 valid acc 16/16
Epoch 24 loss 1.2231267038894893 valid acc 16/16
Epoch 24 loss 2.2169224454975476 valid acc 15/16
Epoch 24 loss 1.118071170493985 valid acc 16/16
Epoch 24 loss 1.857739927530195 valid acc 14/16
Epoch 24 loss 1.3835008439439704 valid acc 15/16
Epoch 24 loss 1.1829166019476818 valid acc 15/16
Epoch 24 loss 0.49700069640941624 valid acc 15/16
Epoch 24 loss 0.4910115876079855 valid acc 16/16
Epoch 24 loss 0.29876455184119244 valid acc 16/16
Epoch 24 loss 0.7763242911793775 valid acc 16/16
Epoch 24 loss 0.7570387496389763 valid acc 16/16
Epoch 24 loss 0.45894426121817256 valid acc 16/16
Epoch 24 loss 1.3148191088379424 valid acc 16/16
Epoch 24 loss 1.0790274794927206 valid acc 16/16
Epoch 24 loss 0.24590455877953912 valid acc 16/16
Epoch 24 loss 0.8973474049730878 valid acc 16/16
Epoch 24 loss 1.2538129422662536 valid acc 16/16
Epoch 24 loss 0.735366156018319 valid acc 15/16
Epoch 24 loss 0.7732732361291197 valid acc 16/16
Epoch 24 loss 2.6118471686954434 valid acc 16/16
Epoch 24 loss 1.1695960159168368 valid acc 15/16
Epoch 24 loss 0.9828858207732512 valid acc 16/16
Epoch 24 loss 0.7723861433611627 valid acc 16/16
Epoch 24 loss 0.9509407708364805 valid acc 16/16
Epoch 24 loss 1.4108301518798698 valid acc 16/16
Epoch 24 loss 0.8841553948951992 valid acc 16/16
Epoch 24 loss 0.7771545145597827 valid acc 16/16
Epoch 24 loss 0.9683583776884681 valid acc 16/16
Epoch 24 loss 0.6811048360338652 valid acc 16/16
Epoch 24 loss 1.4260750577733918 valid acc 14/16
Epoch 24 loss 1.1320292660038642 valid acc 14/16
Epoch 24 loss 0.5742922690039887 valid acc 15/16
Epoch 24 loss 0.9287536192864518 valid acc 16/16
Epoch 24 loss 1.1475150795635074 valid acc 16/16
Epoch 24 loss 0.6571401084206043 valid acc 16/16
Epoch 24 loss 0.6961057559698931 valid acc 16/16
Epoch 24 loss 0.6803246761564985 valid acc 16/16
Epoch 24 loss 1.0424947215532192 valid acc 16/16
Epoch 24 loss 0.6007964529835408 valid acc 16/16
Epoch 24 loss 1.6012187986072435 valid acc 14/16
Epoch 24 loss 0.35880238883283333 valid acc 16/16
Epoch 24 loss 0.9950940898873093 valid acc 16/16
Epoch 24 loss 1.0975179752039328 valid acc 14/16
Epoch 24 loss 0.7658200635601871 valid acc 15/16
Epoch 24 loss 1.1551895269461205 valid acc 16/16
Epoch 24 loss 1.7849186749412662 valid acc 16/16
Epoch 24 loss 1.1688079525567452 valid acc 16/16
Epoch 24 loss 0.879288034648491 valid acc 16/16
Epoch 25 loss 0.17839939613424893 valid acc 16/16
Epoch 25 loss 1.6537976074181118 valid acc 16/16
Epoch 25 loss 1.9891785610681292 valid acc 16/16
Epoch 25 loss 1.2881580499983352 valid acc 16/16
Epoch 25 loss 0.5447752134421724 valid acc 16/16
Epoch 25 loss 0.8835396952223016 valid acc 16/16
Epoch 25 loss 1.8771378764550906 valid acc 16/16
Epoch 25 loss 1.1435681815016334 valid acc 16/16
Epoch 25 loss 1.4979186392480834 valid acc 16/16
Epoch 25 loss 0.6511412954682376 valid acc 16/16
Epoch 25 loss 1.0602562035758092 valid acc 16/16
Epoch 25 loss 1.6705440263276283 valid acc 15/16
Epoch 25 loss 1.3765263652750126 valid acc 16/16
Epoch 25 loss 1.668284448429056 valid acc 16/16
Epoch 25 loss 2.6736180910310545 valid acc 15/16
Epoch 25 loss 1.101796476437603 valid acc 14/16
Epoch 25 loss 1.833770211532669 valid acc 16/16
Epoch 25 loss 0.6950374130137673 valid acc 16/16
Epoch 25 loss 1.8724641021241033 valid acc 15/16
Epoch 25 loss 1.7186344714361634 valid acc 16/16
Epoch 25 loss 0.9800779371608359 valid acc 15/16
Epoch 25 loss 0.7363145870255073 valid acc 16/16
Epoch 25 loss 0.71615625398406 valid acc 15/16
Epoch 25 loss 0.28228760917204576 valid acc 15/16
Epoch 25 loss 0.9974701632175123 valid acc 15/16
Epoch 25 loss 1.1272898045752187 valid acc 15/16
Epoch 25 loss 0.7576852541988687 valid acc 16/16
Epoch 25 loss 0.9065155579734431 valid acc 16/16
Epoch 25 loss 1.0680898273591923 valid acc 15/16
Epoch 25 loss 0.6425834094575804 valid acc 16/16
Epoch 25 loss 1.1841272082058671 valid acc 16/16
Epoch 25 loss 1.0417735492014613 valid acc 16/16
Epoch 25 loss 0.9810964628296617 valid acc 16/16
Epoch 25 loss 1.3081679071173822 valid acc 16/16
Epoch 25 loss 1.7100403209566126 valid acc 16/16
Epoch 25 loss 0.9300639043941124 valid acc 16/16
Epoch 25 loss 0.8462546559864322 valid acc 16/16
Epoch 25 loss 1.221968815671071 valid acc 15/16
Epoch 25 loss 1.4725355165136236 valid acc 16/16
Epoch 25 loss 1.420410429395729 valid acc 16/16
Epoch 25 loss 0.7284667139990091 valid acc 16/16
Epoch 25 loss 1.1456403580461534 valid acc 16/16
Epoch 25 loss 1.3377240911188184 valid acc 15/16
Epoch 25 loss 0.6935443422162949 valid acc 16/16
Epoch 25 loss 1.044298342578644 valid acc 16/16
Epoch 25 loss 0.6161346581002702 valid acc 16/16
Epoch 25 loss 0.755735688640888 valid acc 16/16
Epoch 25 loss 0.9210028573375163 valid acc 16/16
Epoch 25 loss 1.2804356958807055 valid acc 16/16
Epoch 25 loss 0.6799312989338457 valid acc 16/16
Epoch 25 loss 0.5622839831442719 valid acc 16/16
Epoch 25 loss 0.6541826987594138 valid acc 16/16
Epoch 25 loss 1.1316365200444647 valid acc 16/16
Epoch 25 loss 0.39145548820104403 valid acc 16/16
Epoch 25 loss 1.2284460066231657 valid acc 15/16
Epoch 25 loss 0.6463782183107885 valid acc 16/16
Epoch 25 loss 0.9736339871692299 valid acc 16/16
Epoch 25 loss 1.2735518863797162 valid acc 15/16
Epoch 25 loss 1.234466846289732 valid acc 16/16
Epoch 25 loss 1.2253622779657911 valid acc 16/16
Epoch 25 loss 1.3815158840837527 valid acc 15/16
Epoch 25 loss 0.3537829836891384 valid acc 16/16
Epoch 25 loss 1.412489251319554 valid acc 16/16
```
