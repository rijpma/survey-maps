library("terra")

# find masks with positive predictions
masks = list.files(path = "india_tiles_inference_m1/14", pattern = "*mask*", recursive = TRUE, full.name = TRUE)
masks_info = file.info(masks)
masks_with_roads = rownames(masks_info[masks_info$size > 143, ])

set.seed(141414)
masks_with_roads = sample(masks_with_roads, 200)

scans = gsub("india_tiles_inference_m1", "india_tiles", masks_with_roads)
scans = gsub("_mask", "", scans)
overlays = gsub("mask", "overlay", masks_with_roads)

# lb = lapply(tiles_with_roads, \(x) focal(rast(x), w = k2, na.rm = TRUE))

k1 = matrix(1, nrow = 3, ncol = 3)
k2 = matrix(1, nrow = 5, ncol = 5)
k3 = matrix(1, nrow = 7, ncol = 7)
k4 = matrix(1, nrow = 9, ncol = 9)
k5 = matrix(1, nrow = 11, ncol = 11)

pdf("img/masks_and_buffers_m1.pdf", width = 9)
par(mfrow = c(2, 3))
for (i in 1:length(masks_with_roads)){

    mask = masks_with_roads[[i]]

    out_file = gsub("^india_tiles_inference", "india_tiles_inference_buffered", mask)
    out_path = dirname(out_file)

    plot(rast(scans[[i]], noflip = TRUE))
    plot(rast(overlays[[i]], noflip = TRUE))
    title(main = gsub("india_tiles_inference_m1/14", "", mask), cex = 0.8, font.main = 1)

    mask = rast(mask, noflip = TRUE)
    mask_buf_1 = focal(mask, w = k1, max, na.rm = TRUE)
    mask_buf_3 = focal(mask, w = k3, max, na.rm = TRUE)
    mask_buf_5 = focal(mask, w = k5, max, na.rm = TRUE)

    plot(mask, main = "no buffer")
    plot(mask_buf_1, main = "1 pixel buffer")
    plot(mask_buf_3, main = "3 pixel buffer")
    plot(mask_buf_5, main = "5 pixel buffer")

    # if (!file.exists(out_path)) {
    #     dir.create(out_path, recursive = TRUE)
    # }
    # writeRaster(out, out_file)
    #

    cat(i, " ")

}
dev.off()
