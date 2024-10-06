use clap::{arg, value_parser, Command};
use std::io::{BufReader, Read};

struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    weight_momentum: Vec<f32>,
    bias_momentum: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

struct Network {
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    lr: f32,
    momentum: f32,
    hidden: Layer,
    output: Layer,
}

struct InputData {
    images: Vec<u8>,
    labels: Vec<u8>,
    n_images: usize,
    image_size: usize,
}

fn softmax(x: &mut Vec<f32>) {
    let mut max = x[0];
    for i in 1..x.len() {
        if x[i] > max {
            max = x[i];
        }
    }

    let mut sum = 0.0;
    for i in 0..x.len() {
        x[i] = (x[i] - max).exp();
        sum += x[i];
    }

    for i in 0..x.len() {
        x[i] /= sum;
    }
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Layer {
        let n = input_size * output_size;
        let scale = f32::sqrt(2.0 / input_size as f32);
        let mut weights: Vec<f32> = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push((rand::random::<f32>() - 0.5) * 2.0 * scale);
        }

        Layer {
            weights: weights,
            biases: vec![0.0f32; output_size],
            weight_momentum: vec![0.0f32; n],
            bias_momentum: vec![0.0f32; output_size],
            input_size,
            output_size,
        }
    }

    fn forward(&self, input: &Vec<f32>, output: &mut Vec<f32>) {
        for i in 0..self.output_size {
            output[i] = self.biases[i];
        }

        for i in 0..self.input_size {
            let in_j = input[i];
            let weight_row_index = i * self.output_size;
            for j in 0..self.output_size {
                output[j] += in_j * self.weights[weight_row_index + j];
            }
        }

        for i in 0..self.output_size {
            output[i] = if output[i] > 0.0 { output[i] } else { 0.0 };
        }
    }

    fn backward(
        &mut self,
        input: &Vec<f32>,
        output_gradient: &Vec<f32>,
        input_gradient: &mut Option<&mut Vec<f32>>,
        lr: f32,
        momentum: f32,
    ) {
        if let Some(input_grad) = input_gradient {
            for i in 0..self.input_size {
                input_grad[i] = 0.0;
                let weight_row_index = i * self.output_size;
                for j in 0..self.output_size {
                    input_grad[j] += output_gradient[j] * self.weights[weight_row_index + j];
                }
            }
        }

        for i in 0..self.input_size {
            let in_j = input[i];
            let row_index = i * self.output_size;
            for j in 0..self.output_size {
                let grad = output_gradient[j] * in_j;
                self.weight_momentum[row_index + j] =
                    momentum * self.weight_momentum[row_index + j] + lr * grad;
                self.weights[row_index + j] -= self.weight_momentum[row_index + j];

                if let Some(input_grad) = input_gradient {
                    input_grad[i] += output_gradient[j] * self.weights[row_index + j];
                }
            }
        }

        for i in 0..self.output_size {
            self.bias_momentum[i] = momentum * self.bias_momentum[i] + lr * output_gradient[i];
            self.biases[i] -= self.bias_momentum[i];
        }
    }
}

impl Network {
    fn new(
        intput_size: usize,
        output_size: usize,
        hidden_size: usize,
        lr: f32,
        momentum: f32,
    ) -> Network {
        Network {
            input_size: intput_size,
            output_size: output_size,
            hidden_size: hidden_size,
            lr: lr,
            momentum: momentum,
            hidden: Layer::new(intput_size, hidden_size),
            output: Layer::new(hidden_size, output_size),
        }
    }

    fn train(&mut self, input: &Vec<f32>, label: usize) -> Vec<f32> {
        let mut final_output  = vec![0.0f32;self.output_size];
        let mut hidden_output = vec![0.0f32;self.hidden_size];
        let mut output_gradient = vec![0.0f32;self.output_size];
        let mut hidden_gradient = vec![0.0f32;self.hidden_size];

        self.hidden.forward(input, &mut hidden_output);
        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        for i in 0..self.output_size {
            output_gradient[i] = final_output[i] - if i == label { 1.0 } else { 0.0 };
        }

        self.output.backward(
            &hidden_output,
            &output_gradient,
            &mut Some(&mut hidden_gradient),
            self.lr,
            self.momentum,
        );

        for i in 0..self.hidden_size {
            hidden_gradient[i] *= if hidden_output[i] > 0.0 { 1.0 } else { 0.0 };
        }

        self.hidden.backward(
            &input,
            &hidden_gradient,
            &mut Option::None,
            self.lr,
            self.momentum,
        );
        final_output
    }

    fn predict(&self, input: &Vec<f32>) -> usize {
        let mut hidden_output = vec![0.0f32;self.hidden_size];
        let mut final_output = vec![0.0f32;self.output_size];

        self.hidden.forward(input, &mut hidden_output);
        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        let mut max_index = 0;
        for i in 1..self.output_size {
            if final_output[i] > final_output[max_index] {
                max_index = i;
            }
        }

        max_index
    }
}

impl InputData {
    pub fn new(image_filename: &str, label_filename: &str) -> Result<InputData, std::io::Error> {
        let (n_images, rows, cols, images) = InputData::read_mnist_images(image_filename)?;
        let (n_labels, labels) = InputData::read_mnist_labels(label_filename)?;

        if n_images != n_labels {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "number of images and labels must match",
            ));
        }

        let mut input_data = InputData {
            images: images,
            labels: labels,
            n_images: n_images,
            image_size: rows * cols,
        };

        input_data.shuffle_data();

        Ok(input_data)
    }

    fn read_mnist_images(filename: &str) -> Result<(usize, usize, usize, Vec<u8>), std::io::Error> {
        let mut file = std::fs::File::open(filename)?;
        let mut buf = BufReader::new(&mut file);

        let mut buffer = [0u8; 4];
        buf.read_exact(&mut buffer)?;
        buf.read_exact(&mut buffer)?;
        let n_images = u32::from_be_bytes(buffer);

        buf.read_exact(&mut buffer)?;
        let rows = u32::from_be_bytes(buffer);

        buf.read_exact(&mut buffer)?;
        let cols = u32::from_be_bytes(buffer);

        let mut images = Vec::<u8>::with_capacity((n_images * rows * cols) as usize);
        buf.read_to_end(&mut images)?;

        println!("read {} images from {}", n_images, filename);

        Ok((n_images as usize, rows as usize, cols as usize, images))
    }

    fn read_mnist_labels(filename: &str) -> Result<(usize, Vec<u8>), std::io::Error> {
        let mut file = std::fs::File::open(filename)?;
        let mut file = BufReader::new(&mut file);

        let mut buffer = [0u8; 4];
        file.read_exact(&mut buffer)?;
        file.read_exact(&mut buffer)?;
        let n_labels = u32::from_be_bytes(buffer);

        println!("read {} labels from {}", n_labels, filename);

        let mut labels = Vec::<u8>::with_capacity(n_labels as usize);
        file.read_to_end(&mut labels)?;
        Ok((n_labels as usize, labels))
    }

    fn shuffle_data(&mut self) {
        for i in 0..self.n_images {
            let j = rand::random::<usize>() % (i + 1);
            for k in 0..self.image_size {
                let temp = self.images[i * self.image_size + k];
                self.images[i * self.image_size + k] = self.images[j * self.image_size + k];
                self.images[j * self.image_size + k] = temp;
            }
            let temp = self.labels[i];
            self.labels[i] = self.labels[j];
            self.labels[j] = temp;
        }
    }
}

fn main() -> Result<(), std::io::Error> {
    let matches = Command::new("nn")
        .version("1.0")
        .author("fengli <neolidy@foxmail.com>")
        .about("sample rust mnist trainer and predictor.")
        .arg(
            arg!(
                -t --train_file <TRAIN_FILE> "set the train file."
            )
            .required(true),
        )
        .arg(
            arg!(
                -l --label_file <LABEL_FILE> "set the label file."
            )
            .required(true),
        )
        .arg(
            arg!(-L --lr <LEARNING_RATE> "set the learning rate.")
                .required(false)
                .value_parser(value_parser!(f32))
                .default_value("0.0005"),
        )
        .arg(
            arg!(-M --momentum <MOMENTUM> "set the momentum.")
                .required(false)
                .value_parser(value_parser!(f32))
                .default_value("0.9"),
        )
        .arg(
            arg!(-e --epochs <EPOCHS> "set the epochs.")
                .required(false)
                .value_parser(value_parser!(usize))
                .default_value("64"),
        )
        .get_matches();
    let train_file = matches
        .get_one::<String>("train_file")
        .expect("train file is required");
    let label_file = matches
        .get_one::<String>("label_file")
        .expect("label file is required");
    let lr = *matches
        .get_one::<f32>("lr")
        .expect("learning rate is required");
    let momentum = *matches
        .get_one::<f32>("momentum")
        .expect("momentum is required");
    let epochs = *matches
        .get_one::<usize>("epochs")
        .expect("epochs is required");

    let input_data = InputData::new(train_file, label_file)?;
    let mut network = Network::new(input_data.image_size, 10, 100, lr, momentum);

    let train_size = (input_data.n_images as f32 * 0.8) as usize;
    let test_size = input_data.n_images - train_size;
    
    println!("train size: {}, test size: {}", train_size, test_size);
    println!("start training...");
    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        for i in 0..train_size {
            let image = &input_data.images
                [i * input_data.image_size..(i + 1) * input_data.image_size]
                .iter()
                .map(|x| *x as f32 / 255.0)
                .collect();
            let final_output = network.train(image, input_data.labels[i] as usize);
            total_loss += (final_output[input_data.labels[i] as usize] as f32 + 1e-10f32).log10();
        }

        let mut correct = 0;
        for i in train_size..input_data.n_images {
            let image = &input_data.images
                [i * input_data.image_size..(i + 1) * input_data.image_size]
                .iter()
                .map(|x| *x as f32 / 255.0)
                .collect();
            let pred = network.predict(image);
            correct += if pred == input_data.labels[i] as usize {
                1
            } else {
                0
            };
        }

        println!(
            "epoch: {}, loss: {}, accuracy: {}",
            epoch,
            total_loss,
            correct as f32 / test_size as f32
        );
    }

    Ok(())
}
