# AI Use Disclosure

I used OpenAI Codex while completing this assignment.

The AI was used to:

- explain the forward alpha-compositing process and its backward derivatives;
- help derive the colour and alpha adjoints in `render_backward`;
- explain the use of `wp.grad(alpha_at_pixel)` and atomic gradient accumulation;
- help transfer and verify the same gradient logic in `render_sparse_backward`;
- guide environment setup, debugging, testing, training, and evaluation;
- create a temporary dense/sparse gradient-equivalence test.

I verified the resulting implementation using:

- the provided finite-difference gradient checker on CPU;
- the provided finite-difference gradient checker on CUDA;
- a full-coverage comparison between dense and sparse loss and gradient buffers;
- a 2,000-iteration GPU training run;
- evaluation on all 200 held-out test views.

The CPU and CUDA gradient checks both passed. Dense and full-coverage sparse gradients agreed to floating-point precision. The final test-set evaluation reported a mean PSNR of 24.23 dB and a mean SSIM of 0.89222.

I reviewed the final code and remain responsible for understanding and defending the submitted implementation.