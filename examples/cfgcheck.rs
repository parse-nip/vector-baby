fn main() {
    println!("avx512f = {}", cfg!(target_feature = "avx512f"));
    println!("avx2    = {}", cfg!(target_feature = "avx2"));
    println!("fma     = {}", cfg!(target_feature = "fma"));
}
