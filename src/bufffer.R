# find masks with positive predictions
l = list.files(path = "india_tiles_inference/14", pattern = "*mask*", recursive = TRUE, full.names = TRUE)
l_info = file.info(l)

l_info[l_info$size > 143, ]

library("terra")
r = terra::rast("india_tiles_inference/14/12451/6921_lat_26.88288_lon_93.58154_mask.png")
# r[r > 0] = 1

# buffer works with distances which we could convert using res() but let's try focal firt
# 5x5 ones is
k = matrix(1, nrow = 5, ncol = 5)

par(mfrow = c(2,2))
plot(r)
focal(r, w = k, fun = max, na.rm = TRUE) |> plot()
focal(r, w = k, fun = max, na.rm = TRUE) |> plot()
focal(r, w = k, fun = max, na.rm = TRUE) |> plot()
