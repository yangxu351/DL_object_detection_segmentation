
```
数据目录 /data/users/yang/data/
├── synthetic_data_wdt
    ├── {cmt} (syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40_annos)
       ├── {cmt}_images (rgb images, *.jpg)
       ├── {cmt}_annos  (segmentations, *.jpg)
       └── {cmt}_dilated (finetuned segmentations, *.jpg)
    ├── {cmt}_txt_xcycwh 
        ├── minr10_linkr10_px15whr4_all_annos_txt (*.txt)
    ├── {cmt}_xml_annos
        ├── minr10_linkr10_px12whr5_all_xml_annos (*.xml)
    ├── {cmt}_gt_bbox_xcycwh
        ├── minr10_linkr10_px12whr5_all_annos_with_bbox (with bbox *.jpg)
        └── minr10_linkr10_px12whr5_all_images_with_bbox (with bbox *.jpg)
```


```
工作目录 /data/users/yang/code/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn
├── syn_wdt_vockit
    ├── {cmt} (syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40_annos)
       ├── Main
            ├── train.txt
            └── val.txt
    ├── readme.md
    
```