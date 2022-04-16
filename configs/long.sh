
for epochs in 10 50 100 200 500 1000; do
    for lr in 0.001 0.0001 0.00001 0.00001; do
	cat > best12-8a-$epochs-$(echo $lr |tr "." "_").json <<EOF

{
  "dataset": "wf_swinfrared12-8a-4",
  "img_input_shape": [
    64,
    64,
    3
  ],
  "number_of_classes": 2,
  "lr1": $lr,
  "epochs": $epochs,
  "dropout":0.1,
  "hidden_units": 128
}
EOF
    done
done
