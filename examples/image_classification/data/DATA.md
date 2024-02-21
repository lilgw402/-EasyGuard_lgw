# Orginize your own data

Expected datasets structure:

```
imagenet-r
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|  |_ ...
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Here is an example of ImageNet-R

`wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar && tar -xf imagenet-r.tar`

run "bash [`make_data.sh`](make_data.sh) to generate list files
 
