#!/bin/bash


#class cfg:
#    img_input_shape= (64,64,3) # 224,224(224, 224)
#    number_of_classes = 2
#    lr1 = np.random.choice([0.01, 0.001, 0.0001])
#    epochs=np.random.choice([10,25, 50,100])
#    hidden_units = np.random.choice([32,64,128,256,512,1024])
#    dropout = np.random.choice([0.1,0.2,0.5])





for dataset in wf_12-11-04 wf_agriculture11-8-2 wf_colorinfrared_8_4_3 wf_natural_4_3_2 wf_swinfrared12-8a-4; do
    for lr in 0.01 0.001 0.0001 1e-7; do
	for hidden in 32 64 128 256 512 1024; do
cat > $dataset-$(echo $lr |tr "." "_")-$hidden-50ep.conf <<EOF
{
"dataset": "$dataset",
"img_input_shape": [64,64,3],
"number_of_classes": 2,
"lr1": $lr,
"epochs":50,
"hidden_units": $hidden,
"dropout": 0.1
}
EOF
    done
done
done


#for LR in 0.1 0.001 0.00001 1e-7; do
#    default_config | jq ". + {"lr1": $LR}" > $(echo "LR:$LR" |md5sum | cut -b -10).json
#done
