SRC_DIR := "src"
STATIC_DIR := SRC_DIR / "static"
OUT_DIR := "out"
DEV_PORT := "3000"

COMMON_ESBUILD_FLAGS := "--outdir=" + OUT_DIR / "js --bundle --platform=browser --loader:.wgsl=text --format=esm"

_default:
    @just --list

install:
    @bun install

_prepare:
    mkdir -p {{OUT_DIR}}
    cp -r {{STATIC_DIR}}/* {{OUT_DIR}}

dev: clean _prepare
    bunx esbuild {{SRC_DIR}}/*.ts {{COMMON_ESBUILD_FLAGS}} --sourcemap --serve={{DEV_PORT}} --servedir={{OUT_DIR}}

build: clean _prepare
    bunx esbuild {{SRC_DIR}}/*.ts {{COMMON_ESBUILD_FLAGS}} --minify

serve:
    bunx serve {{OUT_DIR}} -p {{DEV_PORT}}

test:
    @bun test

clean:
    rm -rf {{OUT_DIR}}
