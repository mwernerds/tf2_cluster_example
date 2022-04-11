#!/bin/bash


#class cfg:
#    img_input_shape= (64,64,3) # 224,224(224, 224)
#    number_of_classes = 2
#    lr1 = np.random.choice([0.01, 0.001, 0.0001])
#    epochs=np.random.choice([10,25, 50,100])
#    hidden_units = np.random.choice([32,64,128,256,512,1024])
#    dropout = np.random.choice([0.1,0.2,0.5])
function default_config()
{
cat <<EOF
{
"img_input_shape": [64,64,3],
"number_of_classes": 2,
"lr1": 0.001,
"epochs":10,
"hidden_units": 512,
"dropout": 0.1
}
EOF
}

for LR in 0.1 0.001 0.00001 1e-7; do
    default_config | jq ". + {"lr1": $LR}" > $(echo "LR:$LR" |md5sum | cut -b -10).json
done
