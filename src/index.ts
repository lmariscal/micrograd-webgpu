import { Model } from "./nn";

async function main() {
    console.log("Fetching mnist_train.bin");
    const res = await fetch("/mnist_train.bin");
    const buffer = await res.arrayBuffer();
    const u8 = new Uint8Array(buffer);
    console.log("Fetched mnist_train.bin"); 

    const model = new Model([784, 16, 16, 10]);
}

main();
