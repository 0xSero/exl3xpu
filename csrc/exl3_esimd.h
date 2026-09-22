#pragma once
// EXL3 (exllamav3 trellis) kernels for Intel Xe2 (BMG / Arc B70) in ESIMD.
//
// Trellis tile (16x16 weights, K bits each) = 256*K bits = 8*K uint32 words, MSB-first stream.
// Value t (0..255) is decoded from the 16-bit window ending at bit (t+1)*K (circular in the tile),
// and lands at tile row/col:
//     row = 8*((t>>1)&1) + 2*((t>>3)&3) + (t&1)       (K / input dim)
//     col = (t>>5) + 8*((t>>2)&1)                       (N / output dim)
//
// Bit periodicity: with g = gcd(K, 32), every D = K/g words hold V = 32/g whole values, so value
// t = V*grp + u always sits at the same bit offsets relative to word D*grp. We vectorise over grp
// (G = 256/V lanes) and fully unroll u. For V >= 8 the output column half h = (u>>2)&1 depends on
// u only and the base column t>>5 depends on grp only, so accumulators are acc[h][grp] and the
// final reduction sums groups of 32/V adjacent lanes.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace exl3 {

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr int cgcd(int a, int b) { return b == 0 ? a : cgcd(b, a % b); }

template <int K> struct Geo {
    static constexpr int g = cgcd(K, 32);
    static constexpr int D = K / g;          // words per period
    static constexpr int V = 32 / g;         // values per period
    static constexpr int G = 256 / V;        // periods per tile (vector lanes)
    static constexpr int WORDS = 8 * K;      // words per tile
    static_assert(V >= 8, "K with gcd(K,32) > 4 (K=8) not supported by this kernel");
};

// ------------------------------------------------------------------------------------------------
// Codebooks: 16-bit state -> fp16 value (returned widened to float), bit-exact with exllamav3.

template <int CB, int N>
ESIMD_INLINE simd<float, N> decode_cb(simd<uint32_t, N> st) {
    if constexpr (CB == 2) {  // mul1
        simd<uint32_t, N> x = st * 0x83DCD12Du;
#ifdef EXL3_NO_DP4A
        simd<uint32_t, N> y = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);
        simd<uint32_t, N> bs = (y & 0xFFFFu) + (y >> 16);
#else
        // byte sum in one instruction: 0 + dot(bytes(x), {1,1,1,1})
        simd<uint32_t, N> bs = dp4a<uint32_t, uint32_t, uint32_t, uint32_t, N>(
            simd<uint32_t, N>(0u), x, simd<uint32_t, N>(0x01010101u));
#endif
        simd<float, N> f = convert<float>(bs);
        // (1024 + bs) * inv + bias; the exact fp16-rounded codebook value differs by < 2^-11 relative
        f = f * 0.00676727294921875f + (1024.0f * 0.00676727294921875f - 10.3828125f);
#ifdef EXL3_EXACT_FP16
        return convert<float>(convert<fp16>(f));
#else
        return f;
#endif
    } else {
        simd<uint32_t, N> x;
        if constexpr (CB == 1) x = st * 0xCBAC1FEDu;
        else x = st * 89226354u + 64248484u;
        x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
        simd<uint16_t, N> lo = convert<uint16_t>(x & 0xFFFFu);
        simd<uint16_t, N> hi = convert<uint16_t>(x >> 16);
        simd<float, N> a = convert<float>(lo.template bit_cast_view<fp16>().read());
        simd<float, N> b = convert<float>(hi.template bit_cast_view<fp16>().read());
        return convert<float>(convert<fp16>(a + b));
    }
}

// state for value u of every period in a tile: words dw [WORDS]
template <int K, int u>
ESIMD_INLINE simd<uint32_t, Geo<K>::G> tile_states(simd<uint32_t, Geo<K>::WORDS> dw) {
    using Gm = Geo<K>;
    constexpr int D = Gm::D, G = Gm::G, W = Gm::WORDS;
    constexpr int e = (u + 1) * K;                       // end bit within period
    constexpr int i1 = (e - 1) / 32;                     // word holding last bit
    constexpr int b0 = e - 16;                           // start bit (may be negative)
    constexpr int i0 = b0 >= 0 ? b0 / 32 : -1;           // word holding first bit
    constexpr int s = (i1 + 1) * 32 - e;                 // right shift aligning window end
    simd<uint32_t, G> B = dw.template select<G, D>(i1);
    simd<uint32_t, G> st;
    if constexpr (i0 == i1) {
        st = B >> s;
    } else {
        simd<uint32_t, G> A;
        if constexpr (i0 >= 0) {
            A = dw.template select<G, D>(i0);
        } else {
            // previous period's last word; lane 0 wraps to the tile's last word
            A.template select<G - 1, 1>(1) = dw.template select<G - 1, D>(D - 1);
            A[0] = dw[W - 1];
        }
        if constexpr (s == 0) st = B;
        else st = (A << (32 - s)) | (B >> s);
    }
    return st & 0xFFFFu;
}

// ------------------------------------------------------------------------------------------------
// Fast Walsh-Hadamard (Sylvester order) of 128 floats, unnormalised

ESIMD_INLINE void fwht128(simd<float, 128>& v) {
#define EXL3_BFLY(H)                                                                     \
    {                                                                                    \
        auto m = v.template bit_cast_view<float, 128 / (2 * H), 2 * H>();               \
        simd<float, 64> a = m.template select<128 / (2 * H), 1, H, 1>(0, 0).read();     \
        simd<float, 64> b = m.template select<128 / (2 * H), 1, H, 1>(0, H).read();     \
        m.template select<128 / (2 * H), 1, H, 1>(0, 0) = a + b;                         \
        m.template select<128 / (2 * H), 1, H, 1>(0, H) = a - b;                         \
    }
    EXL3_BFLY(1) EXL3_BFLY(2) EXL3_BFLY(4) EXL3_BFLY(8) EXL3_BFLY(16) EXL3_BFLY(32) EXL3_BFLY(64)
#undef EXL3_BFLY
}

constexpr float kRsqrt128 = 0.08838834764831845f;

// ------------------------------------------------------------------------------------------------
// had_in: xh[g, m, :] = fp16( H(x[m, :] * suh[g, :]) / sqrt(128) ), one thread per (g, m, 128-block)

template <typename TIn>
struct HadInKernel {
    const TIn* x; const fp16* suh; fp16* xh;
    int M, Kdim, S, x_stride;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int kb_n = Kdim / 128;
        int kb = id % kb_n;
        int m = (id / kb_n) % M;
        int g = id / (kb_n * M);
        if (g >= S) return;
        simd<TIn, 128> xi = block_load<TIn, 128>(x + (size_t)m * x_stride + kb * 128);
        simd<fp16, 128> su = block_load<fp16, 128>(suh + (size_t)g * Kdim + kb * 128);
        simd<float, 128> v = convert<float>(xi) * convert<float>(su);
        // CUDA path rounds x*suh to fp16 before the transform
        v = convert<float>(convert<fp16>(v));
        fwht128(v);
        v *= kRsqrt128;
        // blocked layout xh[g][k/16][m][16]: a tile-row's A block for all M rows is contiguous
        simd<fp16, 128> vh = convert<fp16>(v);
        int kt = Kdim / 16;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            block_store<fp16, 16>(xh + (((size_t)g * kt + kb * 8 + i) * M + m) * 16, vh.template select<16, 1>(i * 16));
    }
};

// ------------------------------------------------------------------------------------------------
// had_out: out[m, nb*128:+128] = svh * H(sum_p part[p, m, :]) / sqrt(128)

template <typename TOut>
struct HadOutKernel {
    const float* part; const fp16* svh; TOut* out;
    int M, N, P, out_stride;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int nb_n = N / 128;
        int nb = id % nb_n;
        int m = id / nb_n;
        if (m >= M) return;
        simd<float, 128> v = 0.0f;
        for (int p = 0; p < P; ++p)
            v += block_load<float, 128>(part + ((size_t)p * M + m) * N + nb * 128);
        fwht128(v);
        simd<fp16, 128> sv = block_load<fp16, 128>(svh + nb * 128);
        v = v * kRsqrt128 * convert<float>(sv);
        block_store<TOut, 128>(out + (size_t)m * out_stride + nb * 128, convert<TOut>(v));
    }
};

// ------------------------------------------------------------------------------------------------
// Fused trellis-decode GEMV/GEMM for small M.
//   part[p, m, n] = sum_{k in split p} xh[shard(n), m, k] * W_inner[k, n]
// Thread = (column strip of NT tiles, K split). Loops over its tile-rows.

template <int K, int CB, int MR, int NT>
struct GemvKernel {
    const fp16* xh;          // [S, M, Kdim]
    const uint32_t* tr;      // [Kdim/16, N/16, 8K]
    const int* shard_of_nb;  // [N/128]
    float* part;             // [P, M, N]
    int M, Kdim, N, tiles_n, rows_per_split, n_strips;

    static constexpr int G = Geo<K>::G;
    static constexpr int V = Geo<K>::V;
    static constexpr int W = Geo<K>::WORDS;
    // accumulator layout: acc[((m * NT + j) * 2 + h) * G + grp]
    static constexpr int ACC = MR * NT * 2 * G;

    template <int u>
    static ESIMD_INLINE void step(simd<uint32_t, NT * W>& words, simd<float, MR * 16>& xr, simd<float, ACC>& acc) {
        if constexpr (u < V) {
            constexpr int h = (u >> 2) & 1;
            // row(grp, u) = R0 + STR * (grp % P): a strided replicate of the 16-row x block
            constexpr int P = V == 8 ? 4 : (V == 16 ? 2 : 1);
            constexpr int STR = V == 8 ? 2 : (V == 16 ? 4 : 1);
            constexpr int R0 = 8 * ((u >> 1) & 1) + 2 * ((u >> 3) & 3) + (u & 1);
            simd<float, MR * G> xv;
#pragma unroll
            for (int m = 0; m < MR; ++m)
                xv.template select<G, 1>(m * G) = xr.template replicate_vs_w_hs<G / P, 0, P, STR>(m * 16 + R0);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                simd<uint32_t, W> dw = words.template select<W, 1>(j * W);
                simd<float, G> v = decode_cb<CB, G>(tile_states<K, u>(dw));
#pragma unroll
                for (int m = 0; m < MR; ++m) {
                    auto a = acc.template select<G, 1>(((m * NT + j) * 2 + h) * G);
                    a += v * xv.template select<G, 1>(m * G);
                }
            }
            step<u + 1>(words, xr, acc);
        }
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int strip = id % n_strips;
        int p = id / n_strips;
        if (p * rows_per_split >= Kdim / 16) return;
        int tile_n0 = strip * NT;
        int shard = shard_of_nb[(tile_n0 * 16) / 128];
        int r0 = p * rows_per_split;
        int r1 = r0 + rows_per_split;
        if (r1 > Kdim / 16) r1 = Kdim / 16;

        simd<float, ACC> acc = 0.0f;
        const fp16* xbase = xh + (size_t)shard * M * Kdim;   // blocked [k/16][M][16]

        for (int r = r0; r < r1; ++r) {
            simd<uint32_t, NT * W> words =
                block_load<uint32_t, NT * W>(tr + ((size_t)r * tiles_n + tile_n0) * W);
            simd<float, MR * 16> xr = 0.0f;
            if (M == MR) {
                xr = convert<float>(block_load<fp16, MR * 16>(xbase + (size_t)r * M * 16));
            } else {
#pragma unroll
                for (int m = 0; m < MR; ++m)
                    if (m < M)
                        xr.template select<16, 1>(m * 16) =
                            convert<float>(block_load<fp16, 16>(xbase + ((size_t)r * M + m) * 16));
            }
            step<0>(words, xr, acc);
        }

        // reduce groups of L = 32/V adjacent lanes -> 8 columns per half
        constexpr int L = 32 / V;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (m >= M) break;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                simd<float, 16> o;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    constexpr int dummy = 0; (void)dummy;
                    int base = ((m * NT + j) * 2 + h) * G;
                    simd<float, 8> s = acc.template select<8, L>(base);
#pragma unroll
                    for (int l = 1; l < L; ++l) s += acc.template select<8, L>(base + l);
                    o.template select<8, 1>(h * 8) = s;
                }
                block_store<float, 16>(part + ((size_t)p * M + m) * N + (tile_n0 + j) * 16, o);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------------
// DPAS (XMX) trellis GEMM for batched decode: one EXL3 16x16 tile == one DPAS B operand
// (K16 x N16, VNNI: element (k, n) at ((k>>1)*16 + n)*2 + (k&1)). Decoding each tile once feeds
// MB/8 DPAS ops, so decode cost is amortised over the whole batch block.
//
// Writing decoded lanes straight into VNNI: with grp = P*a + b (P = 32/V... see below) the value
// (grp, u) lands at VNNI matrix [8][32] row = R(u) + MS*b, col = 16*h(u) + (u&1) + 2*a, a in 0..7.
// Reading the trellis words in (b-major, a-minor) lane order (a transpose folded into the word
// read) lets each b-row be written with one strided region move.

template <int K, int CB, int MB, int NT>
struct DpasKernel {
    const fp16* xh;          // [S, M, Kdim]
    const uint32_t* tr;      // [Kdim/16, N/16, 8K]
    const int* shard_of_nb;  // [N/128]
    float* part;             // [P, M, N]
    int M, Kdim, N, tiles_n, rows_per_split, n_strips, m_blocks;

    static constexpr int G = Geo<K>::G, V = Geo<K>::V, D = Geo<K>::D, W = Geo<K>::WORDS;
    static constexpr int P = V == 8 ? 4 : (V == 16 ? 2 : 1);   // grp period in rows
    static constexpr int A_ = G / P;                            // == 8
    static constexpr int MS = 4 / P;                            // VNNI-row stride per b
    static_assert(A_ == 8, "lane geometry");

    template <int u>
    static ESIMD_INLINE void build(simd<uint32_t, W>& dw, simd<uint32_t, W>& dwprev, simd<fp16, 256>& Bv) {
        if constexpr (u < V) {
            constexpr int e = (u + 1) * K;
            constexpr int i1 = (e - 1) / 32;
            constexpr int b0 = e - 16;
            constexpr int i0 = b0 >= 0 ? b0 / 32 : -1;
            constexpr int s = (i1 + 1) * 32 - e;
            // transpose-read: lane L = 8*b + a  <->  word D*(P*a + b) + i
            simd<uint32_t, G> Bw = dw.template replicate_vs_w_hs<P, D, 8, D * P>(i1);
            simd<uint32_t, G> st;
            if constexpr (i0 == i1) {
                st = Bw >> s;
            } else {
                simd<uint32_t, G> Aw;
                if constexpr (i0 >= 0) Aw = dw.template replicate_vs_w_hs<P, D, 8, D * P>(i0);
                else Aw = dwprev.template replicate_vs_w_hs<P, D, 8, D * P>(0);
                if constexpr (s == 0) st = Bw;
                else st = (Aw << (32 - s)) | (Bw >> s);
            }
            simd<fp16, G> v = convert<fp16>(decode_cb<CB, G>(st & 0xFFFFu));
            constexpr int h = (u >> 2) & 1;
            constexpr int R = 4 * ((u >> 1) & 1) + ((u >> 3) & 3);      // row0(u) / 2
            constexpr int C0 = 16 * h + (u & 1);
            auto Bm = Bv.template bit_cast_view<fp16, 8, 32>();
#pragma unroll
            for (int b = 0; b < P; ++b)
                Bm.template select<1, 1, 8, 2>(R + MS * b, C0) = v.template select<8, 1>(8 * b);
            build<u + 1>(dw, dwprev, Bv);
        }
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int strip = id % n_strips;
        int rest = id / n_strips;
        int mb = rest % m_blocks;
        int p = rest / m_blocks;
        if (p * rows_per_split >= Kdim / 16) return;
        int tile_n0 = strip * NT;
        int shard = shard_of_nb[(tile_n0 * 16) / 128];
        int r0 = p * rows_per_split;
        int r1 = r0 + rows_per_split;
        if (r1 > Kdim / 16) r1 = Kdim / 16;
        int m0 = mb * MB;
        int mrows = M - m0; if (mrows > MB) mrows = MB;

        simd<float, MB * 16 * NT> acc = 0.0f;   // [NT][MB][16]
        const fp16* xbase = xh + (size_t)shard * M * Kdim;   // blocked [k/16][M][16]

        for (int r = r0; r < r1; ++r) {
            simd<uint32_t, NT * W> words =
                block_load<uint32_t, NT * W>(tr + ((size_t)r * tiles_n + tile_n0) * W);
            simd<fp16, MB * 16> Am;
            const fp16* arow = xbase + ((size_t)r * M + m0) * 16;
            if (mrows == MB) {
                Am = block_load<fp16, MB * 16>(arow);
            } else {
                Am = 0;
#pragma unroll
                for (int m8 = 0; m8 < MB; m8 += 8) {
                    if (m8 + 8 <= mrows) Am.template select<128, 1>(m8 * 16) = block_load<fp16, 128>(arow + m8 * 16);
                    else {
#pragma unroll
                        for (int m = 0; m < 8; ++m)
                            if (m8 + m < mrows)
                                Am.template select<16, 1>((m8 + m) * 16) = block_load<fp16, 16>(arow + (m8 + m) * 16);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                simd<uint32_t, W> dw = words.template select<W, 1>(j * W);
                // previous-period last word per period start: dwprev[D*g] = dw[D*g - 1] (wrap)
                simd<uint32_t, W> dwprev;
                dwprev.template select<W - 1, 1>(1) = dw.template select<W - 1, 1>(0);
                dwprev[0] = dw[W - 1];
                simd<fp16, 256> Bv;
                build<0>(dw, dwprev, Bv);
#pragma unroll
                for (int rb = 0; rb < MB / 8; ++rb) {
                    auto c = acc.template select<128, 1>((j * MB + rb * 8) * 16);
                    c = xmx::dpas<8, 8, float, float, fp16, fp16>(
                        simd<float, 128>(c), Bv, simd<fp16, 128>(Am.template select<128, 1>(rb * 128)));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int m = 0; m < MB; ++m)
                if (m < mrows)
                    block_store<float, 16>(part + ((size_t)p * M + m0 + m) * N + (tile_n0 + j) * 16,
                                           acc.template select<16, 1>((j * MB + m) * 16));
    }
};

}  // namespace exl3
