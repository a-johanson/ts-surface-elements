/**
 * Raw SFC32 state.
 */
export type Sfc32State = {
    a: number;
    b: number;
    c: number;
    d: number;
};

/**
 * Small Fast Counting 32-bit pseudo-random number generator.
 *
 * This implementation keeps its hot path in native 32-bit integer operations,
 * which maps well to JavaScript and TypeScript runtimes.
 */
export class Sfc32 {
    private static readonly WARM_UP_ROUNDS = 12;

    private a = 0;
    private b = 0;
    private c = 0;
    private d = 1;

    /**
     * Creates an SFC32 generator seeded from a 64-bit value.
     *
     * @param seed - Unsigned 64-bit seed.
     */
    constructor(seed = 0n) {
        this.seedFromUint64(seed);
    }

    /**
     * Reseeds the generator from a 64-bit value.
     *
     * This follows the PractRand-style setup:
     * `a = 0`, `b = low32(seed)`, `c = high32(seed)`, `d = 1`, then 12 warm-up rounds.
     *
     * @param seed - Unsigned 64-bit seed.
     */
    seedFromUint64(seed: bigint): void {
        const normalizedSeed = BigInt.asUintN(64, seed);

        this.a = 0;
        this.b = Number(normalizedSeed & 0xffff_ffffn) >>> 0;
        this.c = Number((normalizedSeed >> 32n) & 0xffff_ffffn) >>> 0;
        this.d = 1;

        for (let index = 0; index < Sfc32.WARM_UP_ROUNDS; index++) {
            this.nextUint32();
        }
    }

    /**
     * Restores the raw generator state.
     *
     * @param state - Raw state words.
     */
    setState(state: Sfc32State): void {
        this.a = state.a >>> 0;
        this.b = state.b >>> 0;
        this.c = state.c >>> 0;
        this.d = state.d >>> 0;
    }

    /**
     * Returns the raw generator state.
     *
     * @returns Raw state words.
     */
    getState(): Sfc32State {
        return {
            a: this.a,
            b: this.b,
            c: this.c,
            d: this.d,
        };
    }

    /**
     * Generates the next unsigned 32-bit output.
     *
     * @returns Next unsigned 32-bit output.
     */
    nextUint32(): number {
        const sum = (this.a + this.b) >>> 0;
        const result = (sum + this.d) >>> 0;

        this.d = (this.d + 1) >>> 0;
        this.a = (this.b ^ (this.b >>> 9)) >>> 0;
        this.b = (this.c + (this.c << 3)) >>> 0;
        this.c = (((this.c << 21) | (this.c >>> 11)) + result) >>> 0;

        return result;
    }

    /**
     * Generates the next floating-point output in [0, 1).
     *
     * @returns Next floating-point output.
     */
    nextFloat(): number {
        return this.nextUint32() / 0x1_0000_0000;
    }
}

/**
 * Creates an SFC32-backed random callback that returns values in [0, 1).
 *
 * @param seed - Unsigned 64-bit seed.
 * @returns Random callback.
 */
export function createSfc32(seed = 0n): () => number {
    const generator = new Sfc32(seed);
    return () => generator.nextFloat();
}
