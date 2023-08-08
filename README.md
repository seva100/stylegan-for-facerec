# stylegan-for-facerec

Artem Sevastopolsky, Yury Malkov, Nikita Durasov, Luisa Verdoliva, Matthias Nießner. <i>How to Boost Face Recognition with StyleGAN?</i> // International Conference on Computer Vision (ICCV), 2023

### <img align=center src=./docs/images/project.png width='32'/> [Project page](https://seva100.github.io/stylegan-for-facerec) &ensp; <img align=center src=./docs/images/paper.png width='24'/> [Paper](https://arxiv.org/abs/2210.10090) &ensp; <img align=center src=./docs/images/video.png width='24'> [Video](https://www.youtube.com/watch?v=Bsi0RMTdEaI) &ensp;

<img align=center src=./docs/images/teaser.png width='1200'/>

## About

State-of-the-art face recognition systems require huge amounts of labeled training data. Given the priority of privacy in face recognition applications, the data is limited to celebrity web crawls, which have issues such as skewed distributions of ethnicities and limited numbers of identities. On the other hand, the self-supervised revolution in the industry motivates research on adaptation of the related techniques to facial recognition. One of the most popular practical tricks is to augment the dataset by the samples drawn from the high-resolution high-fidelity models (e.g. StyleGAN-like), while preserving the identity. We show that a simple approach based on fine-tuning an encoder for StyleGAN allows to improve upon the state-of-the-art facial recognition and performs better compared to training on synthetic face identities. We also collect large-scale unlabeled datasets with controllable ethnic constitution — AfricanFaceSet-5M (5 million images of different people) and AsianFaceSet-3M (3 million images of different people) and we show that pretraining on each of them improves recognition of the respective ethnicities (as well as also others), while combining all unlabeled datasets results in the biggest performance increase. Our self-supervised strategy is the most useful with limited amounts of labeled training data, which can be beneficial for more tailored face recognition tasks and when facing privacy concerns. Evaluation is provided based on a standard RFW dataset and a new large-scale RB-WebFace benchmark.

<img align=center src=./docs/images/method.png width='1200'/>

## Data

### BUPT, RFW, LFW

* Our method requires [BUPT-BalancedFace](http://www.whdeng.cn/RFW/Trainingdataste.html) as a source of labeled data for the last stage of training. To access the data, please sign the corresponding license agreement from the authors. We suggest choosing the 400x400 version for higher quality.

Additionally, one needs to align the BUPT data after downloading. For this to work, one should first download the source code of [mtcnn-pytorch](https://github.com/polarisZhao/mtcnn-pytorch) in a separate folder, and then execute:

```bash
python facesets/mtcnn_crop_align.py \
    --in_dir <folder, to which BUPT has been downloaded>/<ethnic group -- one of African,Asian,Indian,Caucasian> \
    --out_dir <output folder where cropped and aligned BUPT should be placed>/<ethnic group> \ \
    --mtcnn_pytorch_path <path, to which mtcnn-pytorch source has been downloaded> \
    --n_threads <number of parallel threads on the same GPU>
```

* RFW test dataset can be accessed by the same link as BUPT-BalancedFace after signing the license agreement. To be used with our [face-evolve](https://github.com/ZhaoJ9014/face.evoLVe)-based framework, images in `test/data` folder in `test.tar.gz` need to be converted to bcolz format. Alignment and conversion can be executed with the following snippet:

```bash
python scripts/rfw_crop_align.py \
    --in_dir <folder, to which RFW has been downloaded> \
    --out_dir <output folder where cropped and aligned RFW should be placed> \ \
    --mtcnn_pytorch_path <path, to which mtcnn-pytorch source has been downloaded> \
    --n_threads <number of parallel threads on the same GPU>

cp -r <path to the downloaded RFW/test/txts> <path to the aligned RFW>/test/txts

python scripts/pack_RFW_in_bcolz.py \
    --data_path <path to the aligned RFW -- the one containing "test" as subfolders>/test/data/<ethnic group=> \
    --out_path <path to the output directory that will contain bcolz array as a subfolder and issame numpy array as a file>/test/data/<ethnic group>
```

* All the test sets available in [face-evolve Data Zoo](https://github.com/ZhaoJ9014/face.evoLVe#data-zoo), such as LFW, CALFW, CPLFW, etc., could also be used for evaluation in our framework. See Subsection <b>Stage 3 (face recognition training)</b> for more details. Make sure to sign the corresponding license agreement for the datasets used.

### AfricanFaceSet, AsianFaceSet

We release two prior data collections in a form of links to YouTube videos and downloading & processing scripts. Namely, we publish:

* Youtube IDs of channels, from which the data has been gathered, in `channel_ids`,
* IDs of all videos, from which the data has been gathered, in `video_ids`.

The index can be downloaded by entering your data in the [form](https://forms.gle/kmbuvNgk6aDnNWaW8). We suppose later that it is downloaded to the `facesets` folder in the project root.

One can either download the same collection of videos as we had via:

```bash
python facesets/download_from_list_parallel.py \
    --list_dir <project root>/facesets/video_ids/AfricanFaceSet \
    --out_dir <folder, to which download the video frames> \
    --n_threads <number of parallel threads>
```

(this requires the [pytube](https://github.com/pytube/pytube) dependency)

Larger number of parallel threads makes sense in case enough CPU cores and fast Internet connection is available. The script downloads videos in parallel (at the same time, checks if it can be downloaded from a given system, in a given country, does not have age restrictions, is not a stream, etc.), extracts the frames, and immediately removes the downloaded video files to save space. The script may run for several days since downloading takes time. 

After downloading, images should be cropped and aligned. For this to work, one should first download the source code of [mtcnn-pytorch](https://github.com/polarisZhao/mtcnn-pytorch) in a separate folder, and then execute:

```bash
python facesets/mtcnn_crop_align.py \
    --in_dir <folder, to which the video frames have been downloaded> \
    --out_dir <output folder with cropped and aligned images> \
    --mtcnn_pytorch_path <path, to which mtcnn-pytorch source has been downloaded> \
    --n_threads <number of parallel threads on the same GPU>
```

This procedure yields roughly 5M images in AfricanFaceSet and 3M images in AsianFaceSet on our side. The published lists of video IDs have been collected around January 2022 and have been actualized in July 2023 this code and datasets release, accounting for deleted or restricted videos.

Since we require a .txt file with the list of all images in the Stage 1 and Stage 2 of training, such a list can be compiled via:

```bash
python facesets/make_filelist.py \
    --dataset_path <folder_AfricanSet> [<folder_AsianSet>] \
    --out_path <path to the output filelist>
```

Another option is to collect all videos from the specified channels, including newly released ones. To perform this, please install [scrapetube](https://github.com/dermasmid/scrapetube) and compile new lists of video IDs via:

```bash
python facesets/get_videos_in_channel_scrapetube_batch.py \
    --info_file <project root>/facesets/channel_ids/AfricanFaceSet.txt \
    --out_dir <new folder, to which save video IDs lists>
```

This procedure can yield significantly larger number of images, depending on the time of execution.

The data released under via the [form](https://forms.gle/kmbuvNgk6aDnNWaW8) and requires agreeing to the Terms of Use. Our data is only published in the form of list of YouTube video IDs and does not contain any personally identifiable information. Similarly, no other information regarding the videos, such as the title, author, or people present is contained in the provided files is collected via the scripts provided. We follow the good practice of data released in the similar way. e.g. [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html), [AVSpeech](https://looking-to-listen.github.io/avspeech/), or [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics). We respect the data subjects' or video owners' rights to request their removal from the list of video IDs as per Art. 14 5(b) EU GDPR law. Please reach out to `artem.sevastopolskiy at tum.de` to have your request processed.

### RB-WebFace

Our RB-WebFace benchmark is constructed as a partition of WebFace-42M (a cleaned version of WebFace-260M), specifying positive and negative pairs for identities that were predicted to be likely belonging to the African, Caucasian, Indian, East Asian ethnical groups. This subdivision was first performed via a neural classifier, originally trained on BUPT ethnic group labels, and then refined by leaving only those identities, for which the classifier consistently predicted the same group for more than 80\% of their pictures in WebFace-42M. We release the inference code for the classifier [in this repo](https://github.com/seva100/ethnicity-classifier).

We release the IDs of people under the same [form](https://forms.gle/kmbuvNgk6aDnNWaW8), clustered into each ethnic group in `RB-WebFace partitions/estimated_ethnicities`, and group-wise positive and negative pairs in `RB-WebFace/pairs`. 

* For positive pairs, located in `pos_pairs_samples_*.txt`, each 5 consecutive filenames correspond to the same person, and we take all pairwise combinations of these 5 images and consider them positive pairs.
* For negative pairs, located in `neg_pairs_samples_*.txt`, all filenames correspond to the images of different people, and we take all pairwise combinations of filenames in the file and consider them negative pairs.

To download RB-WebFace images, please sign the corresponding license for WebFace dataset and download it from the original website. 

The testing script `test_RB_Webface.py` can be used to evaluate the quality of the face recognition model trained in Stage 3 on RB-WebFace dataset. It accepts the paths to the samples pairs lists in the format outlined above. See subsection <b>Stage 3 (face recognition training)</b> for the instructions on the script usage.

## Training

### Stage 1 (StyleGAN pretraining)

#### Running

For Stage 1, we run StyleGAN2-ADA implementation from [stylegan2-ada-lightning](https://github.com/nihalsid/stylegan2-ada-lightning) repository. Please follow the instructions from the respective readme to set up the environment. One needs to train it with a config similar to the one provided in `configs/stage_1_config.yaml` where `dataset_path` is substituted according to the folder where prior dataset (the one on which to train StyleGAN2) is saved. (It might be much easier to use `img_list` parameter instead of `dataset_path` to avoid long initialization)

### Stage 2 (pSp encoder pretraining)

Our Stage 2 code is based on the ReStyle Encoder implementation from [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder) repository. Since we make a few changes (e.g. changing the architecture to the specific IR-SE-50 typically used in face recognition or substituting the suitable StyleGAN2 architecture), we include a fork of this repository in the `restyle-encoder` folder. Note that the majority of the code is directly adapted from [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder). One can also use the same repo to also train an encoder using e4e or ReStyle method.

To set up the training environment, please follow the instructions for [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder). The steps for training follow the ones outlined in the original repo:

1. `cd restyle-encoder`
2. Update `configs/paths_config.py` with the data path of the prior dataset.
```
dataset_paths = {
    ...
	'AfrAsianFaceSet': '<path_to_prior_dataset>',
    ...
}
```
You'll also need to update the `'celeba_test_112p'` path, so that there are some test images available (we did not disable the testing in the code). We include some suitable images in `restyle-encoder/dummy-test-data` that you may use.

3. Run the training. On some systems, you may need to do `export CUDA_HOME=...` (with your CUDA path instead of `...`) before running this command.
```bash
python scripts/train_restyle_psp.py \
    --dataset_type=AfrAsianFaceSet \
    --encoder_type=BackboneEncoder \
    --exp_dir=<your save directory> \
    --workers=32 \
    --batch_size=<batch size> \
    --test_batch_size=4 \
    --test_workers=32 \
    --val_interval=5000 \
    --save_interval=10000 \
    --start_from_latent_avg \
    --lpips_lambda=0.8 \
    --l2_lambda=1 \
    --w_norm_lambda=0 \
    --id_lambda=0.0 \
    --input_nc=6 \
    --n_iters_per_batch=1 \
    --output_size=128 \
    --stylegan_weights=<path to stylegan ckpt from Stage 1> \
    --max_steps=2500000 \
    --generator_ada \
    --n_gpus <number of GPUs>
```

Note that `--batch_size` and `--test_batch_size` value must be multiples of the number of GPUs in `--n_gpus` (but they don't have to be equal). Training till `--max_steps=2500000` is not necessary -- we found that much smaller number of iterations can be enough in practice. 

### Stage 3 (face recognition training)

Our Stage 3 code (main code of this repository) is based on [face-evolve](https://github.com/ZhaoJ9014/face.evoLVe) repository, which we found highly universal and very reliable in practice for training face recognition networks. 

To set up the training environment, please follow the instructions for [face-evolve](https://github.com/ZhaoJ9014/face.evoLVe). To train the network in Stage 3, one must provide the path to the labeled dataset and an encoder checkpoint from Stage 2. The paths and training settings are set in config (see the example in `configs/config_BUPT_IR_50_AfrAsian.py`). Importantly, one should provide `DATA_ROOT`, as a folder containing BUPT-BalancedFace and test data (e.g. RFW or LFW) as subfolders, `ENCODER_CHECKPOINT` and `ENCODER_AVG_IMAGE` as the paths to the Stage 2 checkpoint and respectively estimated average image, and `MULTI_GPU`, `GPU_ID`, `NUM_WORKERS`, `BATCH_SIZE` to the desired training configuration. The config `configs/config_BUPT_IR_50_baseline.py` corresponds to the same network without pretraining (i.e. starting from scratch not from the Stage 2 checkpoint).

To start training, run:

```bash
python train.py --config configs/config_BUPT_IR_50_AfrAsian.py
```

(replacing `configs/config_BUPT_IR_50_AfrAsian.py` with the desired config if needed)

By default, logging is done in Weights & Biases and in stdout, and the quality on RFW can be reported there for each epoch or once per several epochs. To enable other testing datasets available in [face-evolve](https://github.com/ZhaoJ9014/face.evoLVe), such as LFW, download them from face-evolve repo and place them in the `DATA_ROOT` directory. Each of the datasets can be enabled in `get_val_data()` function in `util/utils.py` by uncommenting the respective line (or adding a new one for the dataset not listed there). 

To estimate the quality on RFW (and other test datasets, if enabled) for a given checkpoint after training, one can use the script `test_on_RFW.py`:

```bash
python test_RFW.py \
    --config <config corresponding to the model saved in the checkpoint> \
    [--checkpoint <path to the checkpoint to evaluate (if not provided in BACKBONE_RESUME_ROOT in the config specified)>]
```

Note that the quality on RB-WebFace benchmark can be estimated separately by `rb-webface/scripts/test_RB_Webface.py`:

```bash
python rb-webface/scripts/test_RB_Webface.py \
    --data_path <location of WebFace images> \
    --partition_path <project root>/rb-webface/partition/pairs \
    --model_ckpt_path <ckpt of face recognition model from Stage 3 training> \
    --config_name <corresponding config for Stage 3> \
    --cpu_batch_size 1000 \
    --cpu_n_jobs 8 \
    --gpu_batch_size 50
```

Make sure to control CPU batch size, CPU number of jobs and GPU batch size to avoid overflows. The script calculates pairwise distances and therefore can take a long time to run with too few threads. For both `test.py` and `test_on_RFW.py`, not all the values in config are used, but only the ones that are relevant for the evaluation.

Checkpoints for the baseline and pretrained ArcFace R-{34,50,100} models are available [here](https://drive.google.com/drive/folders/1UDxqL5kyGnkncHGY2iq6b2J7pNIWmlET?usp=drive_link).

## Known issues

* Likely due to SGD used as a main optimizer in Stage 3 (both in our work and in [face-evolve](https://github.com/ZhaoJ9014/face.evoLVe)), the ArcFace baseline network sometimes converges to a bad local minimum (one can notice that by loss going on plateau of the value around ~20). Restarting with a different batch size or number of GPUs helped in these cases. The same sometimes happened for other baselines methods or when training on 1% of labeled data. We have never encountered that with a good initialization of the network (e.g. from our Stage 2).

## Licenses

This code is released under MIT License. Since most of the code is based on the contributions from other libraries, such as face-evolve and restyle-encoder, please check the respective licenses for the respective parts of the code.

## Acknowledgments

* We thank the authors of [stylegan2-ada-lightning](https://github.com/nihalsid/stylegan2-ada-lightning), [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder), [face-evolve](https://github.com/ZhaoJ9014/face.evoLVe), [scrapetube](https://github.com/dermasmid/scrapetube), [pytube](https://github.com/pytube/pytube), [mtcnn-pytorch](https://github.com/polarisZhao/mtcnn-pytorch), and other libraries we relied upon, for their effort in creating easy-to-use and reliable codebases.
* We are grateful to the creators of BUPT, RFW, WebFace and other large datasets for face recognition.

## BibTeX

Please cite our work if you found our contributions, code, or datasets useful:

```latex
@article{sevastopolsky2023boost,
  title={How to Boost Face Recognition with StyleGAN?},
  author={Sevastopolsky, Artem and Malkov, Yury and Durasov, Nikita and Verdoliva, Luisa and Nie{\ss}ner, Matthias},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
