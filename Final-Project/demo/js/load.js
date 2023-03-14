function load(element_src) {
    sample_folder = element_src.substring(0, element_src.lastIndexOf('/'));

    sample_folder_name = sample_folder.substring(sample_folder.lastIndexOf('/') + 1, sample_folder.length);

    console.log("Switch to: " + sample_folder);

    document.getElementById("sampletxt").textContent = "Sample: " + sample_folder_name;

    document.getElementById("imgorg").src = sample_folder + "/img.png";
    document.getElementById("imgmasked").src = sample_folder + "/img_masked.png";
    document.getElementById("imgbound").src = sample_folder + "/img_bounded.png";

    document.getElementById("imgsegcolor").src = sample_folder + "/img_seg_color.png";
    document.getElementById("imgsegreg").src = sample_folder + "/img_seg_region.png";
    document.getElementById("imgseglap").src = sample_folder + "/img_seg_lap_edge.png";
    document.getElementById("imgsegsobel").src = sample_folder + "/img_seg_sobel_edge.png";

    document.getElementById("imgear").src = sample_folder + "/img_ear_location.png";
    document.getElementById("imgcolhist").src = sample_folder + "/img_color_hist.png";

    document.getElementById("imgclass").data = sample_folder + "/img_class.txt";
}
