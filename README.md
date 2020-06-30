
## Datasets

### COVID19

- https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM

### COVID19 V2

- https://s3.ca-central-1.amazonaws.com/ubccic.covid19.models/L3netDemoData.zip

### COVID19 V3

- https://zenodo.org/record/3757476#.XtU6wC2ZOuV (create directories: CT, Lung_Mask, Infection_Mask)

### JCU fish

- https://www.dropbox.com/sh/b2jlua76ogyr5rk/AABsJVljG7v2BOunE1k4f_XTa?dl=0

## Active Learning for Counting

```
python trainval.py -e al_trancos -sb <savedir_base> -d <datadir> -r 1
```

## Active Learning for Segmentation

```
python trainval.py -e al_cp -sb <savedir_base> -d <datadir> -r 1
```

## Active Learning for Covid19

```
python trainval.py -e al_covid19_v2 -sb <savedir_base> -d <datadir> -r 1
```
## Weakly supervised for Covid19 version 2

```
python trainval.py -e weakly_covid19_v2_mixed -sb <savedir_base> -d <datadir> -r 1
```

## Weakly supervised for Covid19

```
python trainval.py -e weakly_covid19 -sb <savedir_base> -d <datadir> -r 1
```

## Weakly supervised for JCU fish

```
python trainval.py -e weakly_JCUfish -sb <savedir_base> -d <datadir> -r 1
```

