use rustvision::augment::augment;
use rustvision::checkpoint;
use rustvision::mnist::{Dataset, SIDE};
use rustvision::net::{argmax, softmax, Net};
use rustvision::optim::Adam;
use rustvision::parallel::{num_threads, par_chunks};
use rustvision::rng::Rng;
use std::collections::HashMap;
use std::path::Path;
use std::process::exit;
use std::time::Instant;

const USAGE: &str = "\
rustvision: a CNN built from scratch in pure Rust

USAGE:
  rustvision train [--data data] [--out model.bin] [--epochs 8] [--batch 64]
                   [--lr 2e-3] [--augment true] [--seed 7]
  rustvision eval  [--data data] [--model model.bin]
  rustvision show  [--data data] [--model model.bin] [--count 4] [--seed 1]

Get the data first with scripts/download_mnist.sh.
Set RUSTVISION_THREADS to control parallelism.";

struct Args(HashMap<String, String>);

impl Args {
    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        match self.0.get(key) {
            None => default,
            Some(v) => v.parse().unwrap_or_else(|_| fail(&format!("invalid value for --{key}: {v}"))),
        }
    }

    fn str(&self, key: &str, default: &str) -> String {
        self.0.get(key).cloned().unwrap_or_else(|| default.to_string())
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}\n\n{USAGE}");
    exit(1);
}

fn parse_args() -> (String, Args) {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_else(|| fail("missing command"));
    let mut map = HashMap::new();
    while let Some(k) = it.next() {
        let key = k.strip_prefix("--").unwrap_or_else(|| fail(&format!("unexpected argument {k}")));
        let val = it.next().unwrap_or_else(|| fail(&format!("--{key} needs a value")));
        map.insert(key.to_string(), val);
    }
    (cmd, Args(map))
}

fn main() {
    let (cmd, args) = parse_args();
    match cmd.as_str() {
        "train" => train(&args),
        "eval" => eval(&args),
        "show" => show(&args),
        "help" | "--help" | "-h" => println!("{USAGE}"),
        _ => fail(&format!("unknown command {cmd}")),
    }
}

fn load_data(args: &Args, prefix: &str) -> Dataset {
    let dir = args.str("data", "data");
    Dataset::load(Path::new(&dir), prefix)
        .unwrap_or_else(|e| fail(&format!("could not load MNIST from {dir}/: {e} (run scripts/download_mnist.sh)")))
}

fn load_model(args: &Args) -> Net {
    let path = args.str("model", "model.bin");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| fail(&format!("could not read {path}: {e}")));
    let mut net = Net::mnist();
    checkpoint::load_into(&mut net, &bytes).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    net
}

/// Classifies every test image in parallel. Returns the predicted labels.
fn predict_all(net: &Net, data: &Dataset) -> Vec<usize> {
    let idx: Vec<usize> = (0..data.len()).collect();
    par_chunks(&idx, |chunk| {
        chunk.iter().map(|&i| argmax(net.forward(data.image(i)).last().unwrap())).collect::<Vec<_>>()
    })
    .concat()
}

fn accuracy(net: &Net, data: &Dataset) -> f32 {
    let preds = predict_all(net, data);
    preds.iter().zip(&data.labels).filter(|(p, l)| **p == **l as usize).count() as f32 / data.len() as f32
}

fn train(args: &Args) {
    let out = args.str("out", "model.bin");
    let epochs: usize = args.get("epochs", 8);
    let batch: usize = args.get("batch", 64);
    let max_lr: f32 = args.get("lr", 2e-3);
    let use_aug: bool = args.get("augment", true);
    let mut rng = Rng::new(args.get("seed", 7));

    let train = load_data(args, "train");
    let test = load_data(args, "t10k");
    let mut net = Net::mnist();
    net.init(&mut rng);
    let mut opt = Adam::new(&net);

    println!("data      {} training images, {} test images", train.len(), test.len());
    println!("model     {:?}", net.layers);
    println!("          {} parameters", net.num_params());
    println!(
        "training  {epochs} epochs, batch {batch}, lr {max_lr}, augmentation {}, {} threads\n",
        if use_aug { "on" } else { "off" },
        num_threads()
    );

    let steps_per_epoch = train.len() / batch;
    let total_steps = epochs * steps_per_epoch;
    let mut step = 0;
    let started = Instant::now();
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        let mut order: Vec<usize> = (0..train.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let (mut loss_sum, mut correct, mut seen) = (0.0f64, 0usize, 0usize);
        for (b, idx) in order.chunks_exact(batch).enumerate() {
            // Cosine decay from max_lr to 2% of it over the whole run.
            let lr = max_lr * (0.02 + 0.49 * (1.0 + (std::f32::consts::PI * step as f32 / total_steps as f32).cos()));
            // Each thread gets its own RNG stream for augmentation and its own gradient buffer.
            let seeds: Vec<(u64, &[usize])> =
                idx.chunks(batch.div_ceil(num_threads())).map(|c| (rng.next_u64(), c)).collect();
            let results = par_chunks(&seeds, |work| {
                let mut grads = net.zero_grads();
                let (mut loss, mut ok) = (0.0f64, 0);
                for &(seed, items) in work {
                    let mut trng = Rng::new(seed);
                    for &i in items {
                        let label = train.labels[i] as usize;
                        let img = if use_aug { augment(train.image(i), &mut trng) } else { train.image(i).to_vec() };
                        let (l, pred) = net.train_step(&img, label, &mut grads);
                        loss += l as f64;
                        ok += (pred == label) as usize;
                    }
                }
                (grads, loss, ok)
            });
            let mut grads = net.zero_grads();
            for (g, l, ok) in results {
                for (a, b) in grads.iter_mut().zip(&g) {
                    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                }
                loss_sum += l;
                correct += ok;
            }
            seen += idx.len();
            opt.step(&mut net.params, &grads, 1.0 / batch as f32, lr);
            step += 1;

            if (b + 1) % 100 == 0 {
                println!(
                    "epoch {epoch} | batch {:4}/{steps_per_epoch} | loss {:.4} | train acc {:5.2}% | lr {lr:.2e} | {:.0} img/s",
                    b + 1,
                    loss_sum / seen as f64,
                    100.0 * correct as f32 / seen as f32,
                    seen as f32 / epoch_start.elapsed().as_secs_f32()
                );
            }
        }
        let acc = accuracy(&net, &test);
        println!(
            "\n=== epoch {epoch}: test accuracy {:.2}% ({} / {} wrong), {:.1}s\n",
            100.0 * acc,
            ((1.0 - acc) * test.len() as f32).round(),
            test.len(),
            epoch_start.elapsed().as_secs_f32()
        );
        std::fs::write(&out, checkpoint::to_bytes(&net))
            .unwrap_or_else(|e| fail(&format!("could not save {out}: {e}")));
    }
    println!("done in {:.1} min, model saved to {out}", started.elapsed().as_secs_f32() / 60.0);
}

fn eval(args: &Args) {
    let net = load_model(args);
    let test = load_data(args, "t10k");
    let preds = predict_all(&net, &test);
    let mut confusion = [[0usize; 10]; 10];
    for (&p, &l) in preds.iter().zip(&test.labels) {
        confusion[l as usize][p] += 1;
    }
    let correct: usize = (0..10).map(|i| confusion[i][i]).sum();
    println!("test accuracy: {:.2}% ({correct} / {})\n", 100.0 * correct as f32 / test.len() as f32, test.len());
    println!("confusion matrix (rows: true digit, columns: predicted)\n");
    println!("      {}", (0..10).map(|d| format!("{d:>5}")).collect::<String>());
    for (d, row) in confusion.iter().enumerate() {
        let cells: String =
            row.iter().map(|&n| format!("{:>5}", if n == 0 { ".".to_string() } else { n.to_string() })).collect();
        let total: usize = row.iter().sum();
        println!("   {d}  {cells}   {:6.2}%", 100.0 * row[d] as f32 / total as f32);
    }
}

/// Prints random test digits as text art next to the network's top guesses.
fn show(args: &Args) {
    let net = load_model(args);
    let test = load_data(args, "t10k");
    let mut rng = Rng::new(args.get("seed", 1));
    let ramp: Vec<char> = " .:-=+*#%@".chars().collect();
    for _ in 0..args.get("count", 4usize) {
        let i = rng.below(test.len());
        let img = test.image(i);
        let probs = softmax(net.forward(img).last().unwrap());
        let mut order: Vec<usize> = (0..10).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut side: Vec<String> = vec![format!("test image #{i}, true label {}", test.labels[i]), String::new()];
        for &d in order.iter().take(3) {
            side.push(format!("{d}  {:5.1}%  {}", 100.0 * probs[d], "█".repeat((probs[d] * 30.0).round() as usize)));
        }
        for y in 0..SIDE {
            let row: String = (0..SIDE)
                .flat_map(|x| {
                    let c = ramp[(img[y * SIDE + x] * (ramp.len() - 1) as f32).round() as usize];
                    [c, c]
                })
                .collect();
            println!("{row}   {}", side.get(y.saturating_sub(10)).filter(|_| y >= 10).map_or("", |s| s));
        }
        println!();
    }
}
