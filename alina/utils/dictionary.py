pair_dict = {'A/A': 1, 'A/AA': 2, 'A/AG': 3, 'A/AC': 4, 'A/AU': 5, 'A/AN': 6, 'A/G': 7, 'A/GA': 8, 'A/GG': 9, 'A/GC': 10, 'A/GU': 11, 'A/GN': 12, 'A/C': 13, 'A/CA': 14, 'A/CG': 15, 'A/CC': 16, 'A/CU': 17, 'A/CN': 18, 'A/U': 19, 'A/UA': 20, 'A/UG': 21, 'A/UC': 22, 'A/UU': 23, 'A/UN': 24, 'A/N': 0, 'A/NA': 25, 'A/NG': 26, 'A/NC': 27, 'A/NU': 28, 'A/NN': 0, 'AA/A': 29, 'AA/AA': 30, 'AA/AG': 31, 'AA/AC': 32, 'AA/AU': 33, 'AA/AN': 34, 'AA/G': 35, 'AA/GA': 36, 'AA/GG': 37, 'AA/GC': 38, 'AA/GU': 39, 'AA/GN': 40, 'AA/C': 41, 'AA/CA': 42, 'AA/CG': 43, 'AA/CC': 44, 'AA/CU': 45, 'AA/CN': 46, 'AA/U': 47, 'AA/UA': 48, 'AA/UG': 49, 'AA/UC': 50, 'AA/UU': 51, 'AA/UN': 52, 'AA/N': 0, 'AA/NA': 53, 'AA/NG': 54, 'AA/NC': 55, 'AA/NU': 56, 'AA/NN': 0, 'AG/A': 57, 'AG/AA': 58, 'AG/AG': 59, 'AG/AC': 60, 'AG/AU': 61, 'AG/AN': 62, 'AG/G': 63, 'AG/GA': 64, 'AG/GG': 65, 'AG/GC': 66, 'AG/GU': 67, 'AG/GN': 68, 'AG/C': 69, 'AG/CA': 70, 'AG/CG': 71, 'AG/CC': 72, 'AG/CU': 73, 'AG/CN': 74, 'AG/U': 75, 'AG/UA': 76, 'AG/UG': 77, 'AG/UC': 78, 'AG/UU': 79, 'AG/UN': 80, 'AG/N': 0, 'AG/NA': 81, 'AG/NG': 82, 'AG/NC': 83, 'AG/NU': 84, 'AG/NN': 0, 'AC/A': 85, 'AC/AA': 86, 'AC/AG': 87, 'AC/AC': 88, 'AC/AU': 89, 'AC/AN': 90, 'AC/G': 91, 'AC/GA': 92, 'AC/GG': 93, 'AC/GC': 94, 'AC/GU': 95, 'AC/GN': 96, 'AC/C': 97, 'AC/CA': 98, 'AC/CG': 99, 'AC/CC': 100, 'AC/CU': 101, 'AC/CN': 102, 'AC/U': 103, 'AC/UA': 104, 'AC/UG': 105, 'AC/UC': 106, 'AC/UU': 107, 'AC/UN': 108, 'AC/N': 0, 'AC/NA': 109, 'AC/NG': 110, 'AC/NC': 111, 'AC/NU': 112, 'AC/NN': 0, 'AU/A': 113, 'AU/AA': 114, 'AU/AG': 115, 'AU/AC': 116, 'AU/AU': 117, 'AU/AN': 118, 'AU/G': 119, 'AU/GA': 120, 'AU/GG': 121, 'AU/GC': 122, 'AU/GU': 123, 'AU/GN': 124, 'AU/C': 125, 'AU/CA': 126, 'AU/CG': 127, 'AU/CC': 128, 'AU/CU': 129, 'AU/CN': 130, 'AU/U': 131, 'AU/UA': 132, 'AU/UG': 133, 'AU/UC': 134, 'AU/UU': 135, 'AU/UN': 136, 'AU/N': 0, 'AU/NA': 137, 'AU/NG': 138, 'AU/NC': 139, 'AU/NU': 140, 'AU/NN': 0, 'AN/A': 141, 'AN/AA': 142, 'AN/AG': 143, 'AN/AC': 144, 'AN/AU': 145, 'AN/AN': 146, 'AN/G': 147, 'AN/GA': 148, 'AN/GG': 149, 'AN/GC': 150, 'AN/GU': 151, 'AN/GN': 152, 'AN/C': 153, 'AN/CA': 154, 'AN/CG': 155, 'AN/CC': 156, 'AN/CU': 157, 'AN/CN': 158, 'AN/U': 159, 'AN/UA': 160, 'AN/UG': 161, 'AN/UC': 162, 'AN/UU': 163, 'AN/UN': 164, 'AN/N': 0, 'AN/NA': 165, 'AN/NG': 166, 'AN/NC': 167, 'AN/NU': 168, 'AN/NN': 0, 'G/A': 169, 'G/AA': 170, 'G/AG': 171, 'G/AC': 172, 'G/AU': 173, 'G/AN': 174, 'G/G': 175, 'G/GA': 176, 'G/GG': 177, 'G/GC': 178, 'G/GU': 179, 'G/GN': 180, 'G/C': 181, 'G/CA': 182, 'G/CG': 183, 'G/CC': 184, 'G/CU': 185, 'G/CN': 186, 'G/U': 187, 'G/UA': 188, 'G/UG': 189, 'G/UC': 190, 'G/UU': 191, 'G/UN': 192, 'G/N': 0, 'G/NA': 193, 'G/NG': 194, 'G/NC': 195, 'G/NU': 196, 'G/NN': 0, 'GA/A': 197, 'GA/AA': 198, 'GA/AG': 199, 'GA/AC': 200, 'GA/AU': 201, 'GA/AN': 202, 'GA/G': 203, 'GA/GA': 204, 'GA/GG': 205, 'GA/GC': 206, 'GA/GU': 207, 'GA/GN': 208, 'GA/C': 209, 'GA/CA': 210, 'GA/CG': 211, 'GA/CC': 212, 'GA/CU': 213, 'GA/CN': 214, 'GA/U': 215, 'GA/UA': 216, 'GA/UG': 217, 'GA/UC': 218, 'GA/UU': 219, 'GA/UN': 220, 'GA/N': 0, 'GA/NA': 221, 'GA/NG': 222, 'GA/NC': 223, 'GA/NU': 224, 'GA/NN': 0, 'GG/A': 225, 'GG/AA': 226, 'GG/AG': 227, 'GG/AC': 228, 'GG/AU': 229, 'GG/AN': 230, 'GG/G': 231, 'GG/GA': 232, 'GG/GG': 233, 'GG/GC': 234, 'GG/GU': 235, 'GG/GN': 236, 'GG/C': 237, 'GG/CA': 238, 'GG/CG': 239, 'GG/CC': 240, 'GG/CU': 241, 'GG/CN': 242, 'GG/U': 243, 'GG/UA': 244, 'GG/UG': 245, 'GG/UC': 246, 'GG/UU': 247, 'GG/UN': 248, 'GG/N': 0, 'GG/NA': 249, 'GG/NG': 250, 'GG/NC': 251, 'GG/NU': 252, 'GG/NN': 0, 'GC/A': 253, 'GC/AA': 254, 'GC/AG': 255, 'GC/AC': 256, 'GC/AU': 257, 'GC/AN': 258, 'GC/G': 259, 'GC/GA': 260, 'GC/GG': 261, 'GC/GC': 262, 'GC/GU': 263, 'GC/GN': 264, 'GC/C': 265, 'GC/CA': 266, 'GC/CG': 267, 'GC/CC': 268, 'GC/CU': 269, 'GC/CN': 270, 'GC/U': 271, 'GC/UA': 272, 'GC/UG': 273, 'GC/UC': 274, 'GC/UU': 275, 'GC/UN': 276, 'GC/N': 0, 'GC/NA': 277, 'GC/NG': 278, 'GC/NC': 279, 'GC/NU': 280, 'GC/NN': 0, 'GU/A': 281, 'GU/AA': 282, 'GU/AG': 283, 'GU/AC': 284, 'GU/AU': 285, 'GU/AN': 286, 'GU/G': 287, 'GU/GA': 288, 'GU/GG': 289, 'GU/GC': 290, 'GU/GU': 291, 'GU/GN': 292, 'GU/C': 293, 'GU/CA': 294, 'GU/CG': 295, 'GU/CC': 296, 'GU/CU': 297, 'GU/CN': 298, 'GU/U': 299, 'GU/UA': 300, 'GU/UG': 301, 'GU/UC': 302, 'GU/UU': 303, 'GU/UN': 304, 'GU/N': 0, 'GU/NA': 305, 'GU/NG': 306, 'GU/NC': 307, 'GU/NU': 308, 'GU/NN': 0, 'GN/A': 309, 'GN/AA': 310, 'GN/AG': 311, 'GN/AC': 312, 'GN/AU': 313, 'GN/AN': 314, 'GN/G': 315, 'GN/GA': 316, 'GN/GG': 317, 'GN/GC': 318, 'GN/GU': 319, 'GN/GN': 320, 'GN/C': 321, 'GN/CA': 322, 'GN/CG': 323, 'GN/CC': 324, 'GN/CU': 325, 'GN/CN': 326, 'GN/U': 327, 'GN/UA': 328, 'GN/UG': 329, 'GN/UC': 330, 'GN/UU': 331, 'GN/UN': 332, 'GN/N': 0, 'GN/NA': 333, 'GN/NG': 334, 'GN/NC': 335, 'GN/NU': 336, 'GN/NN': 0, 'C/A': 337, 'C/AA': 338, 'C/AG': 339, 'C/AC': 340, 'C/AU': 341, 'C/AN': 342, 'C/G': 343, 'C/GA': 344, 'C/GG': 345, 'C/GC': 346, 'C/GU': 347, 'C/GN': 348, 'C/C': 349, 'C/CA': 350, 'C/CG': 351, 'C/CC': 352, 'C/CU': 353, 'C/CN': 354, 'C/U': 355, 'C/UA': 356, 'C/UG': 357, 'C/UC': 358, 'C/UU': 359, 'C/UN': 360, 'C/N': 0, 'C/NA': 361, 'C/NG': 362, 'C/NC': 363, 'C/NU': 364, 'C/NN': 0, 'CA/A': 365, 'CA/AA': 366, 'CA/AG': 367, 'CA/AC': 368, 'CA/AU': 369, 'CA/AN': 370, 'CA/G': 371, 'CA/GA': 372, 'CA/GG': 373, 'CA/GC': 374, 'CA/GU': 375, 'CA/GN': 376, 'CA/C': 377, 'CA/CA': 378, 'CA/CG': 379, 'CA/CC': 380, 'CA/CU': 381, 'CA/CN': 382, 'CA/U': 383, 'CA/UA': 384, 'CA/UG': 385, 'CA/UC': 386, 'CA/UU': 387, 'CA/UN': 388, 'CA/N': 0, 'CA/NA': 389, 'CA/NG': 390, 'CA/NC': 391, 'CA/NU': 392, 'CA/NN': 0, 'CG/A': 393, 'CG/AA': 394, 'CG/AG': 395, 'CG/AC': 396, 'CG/AU': 397, 'CG/AN': 398, 'CG/G': 399, 'CG/GA': 400, 'CG/GG': 401, 'CG/GC': 402, 'CG/GU': 403, 'CG/GN': 404, 'CG/C': 405, 'CG/CA': 406, 'CG/CG': 407, 'CG/CC': 408, 'CG/CU': 409, 'CG/CN': 410, 'CG/U': 411, 'CG/UA': 412, 'CG/UG': 413, 'CG/UC': 414, 'CG/UU': 415, 'CG/UN': 416, 'CG/N': 0, 'CG/NA': 417, 'CG/NG': 418, 'CG/NC': 419, 'CG/NU': 420, 'CG/NN': 0, 'CC/A': 421, 'CC/AA': 422, 'CC/AG': 423, 'CC/AC': 424, 'CC/AU': 425, 'CC/AN': 426, 'CC/G': 427, 'CC/GA': 428, 'CC/GG': 429, 'CC/GC': 430, 'CC/GU': 431, 'CC/GN': 432, 'CC/C': 433, 'CC/CA': 434, 'CC/CG': 435, 'CC/CC': 436, 'CC/CU': 437, 'CC/CN': 438, 'CC/U': 439, 'CC/UA': 440, 'CC/UG': 441, 'CC/UC': 442, 'CC/UU': 443, 'CC/UN': 444, 'CC/N': 0, 'CC/NA': 445, 'CC/NG': 446, 'CC/NC': 447, 'CC/NU': 448, 'CC/NN': 0, 'CU/A': 449, 'CU/AA': 450, 'CU/AG': 451, 'CU/AC': 452, 'CU/AU': 453, 'CU/AN': 454, 'CU/G': 455, 'CU/GA': 456, 'CU/GG': 457, 'CU/GC': 458, 'CU/GU': 459, 'CU/GN': 460, 'CU/C': 461, 'CU/CA': 462, 'CU/CG': 463, 'CU/CC': 464, 'CU/CU': 465, 'CU/CN': 466, 'CU/U': 467, 'CU/UA': 468, 'CU/UG': 469, 'CU/UC': 470, 'CU/UU': 471, 'CU/UN': 472, 'CU/N': 0, 'CU/NA': 473, 'CU/NG': 474, 'CU/NC': 475, 'CU/NU': 476, 'CU/NN': 0, 'CN/A': 477, 'CN/AA': 478, 'CN/AG': 479, 'CN/AC': 480, 'CN/AU': 481, 'CN/AN': 482, 'CN/G': 483, 'CN/GA': 484, 'CN/GG': 485, 'CN/GC': 486, 'CN/GU': 487, 'CN/GN': 488, 'CN/C': 489, 'CN/CA': 490, 'CN/CG': 491, 'CN/CC': 492, 'CN/CU': 493, 'CN/CN': 494, 'CN/U': 495, 'CN/UA': 496, 'CN/UG': 497, 'CN/UC': 498, 'CN/UU': 499, 'CN/UN': 500, 'CN/N': 0, 'CN/NA': 501, 'CN/NG': 502, 'CN/NC': 503, 'CN/NU': 504, 'CN/NN': 0, 'U/A': 505, 'U/AA': 506, 'U/AG': 507, 'U/AC': 508, 'U/AU': 509, 'U/AN': 510, 'U/G': 511, 'U/GA': 512, 'U/GG': 513, 'U/GC': 514, 'U/GU': 515, 'U/GN': 516, 'U/C': 517, 'U/CA': 518, 'U/CG': 519, 'U/CC': 520, 'U/CU': 521, 'U/CN': 522, 'U/U': 523, 'U/UA': 524, 'U/UG': 525, 'U/UC': 526, 'U/UU': 527, 'U/UN': 528, 'U/N': 0, 'U/NA': 529, 'U/NG': 530, 'U/NC': 531, 'U/NU': 532, 'U/NN': 0, 'UA/A': 533, 'UA/AA': 534, 'UA/AG': 535, 'UA/AC': 536, 'UA/AU': 537, 'UA/AN': 538, 'UA/G': 539, 'UA/GA': 540, 'UA/GG': 541, 'UA/GC': 542, 'UA/GU': 543, 'UA/GN': 544, 'UA/C': 545, 'UA/CA': 546, 'UA/CG': 547, 'UA/CC': 548, 'UA/CU': 549, 'UA/CN': 550, 'UA/U': 551, 'UA/UA': 552, 'UA/UG': 553, 'UA/UC': 554, 'UA/UU': 555, 'UA/UN': 556, 'UA/N': 0, 'UA/NA': 557, 'UA/NG': 558, 'UA/NC': 559, 'UA/NU': 560, 'UA/NN': 0, 'UG/A': 561, 'UG/AA': 562, 'UG/AG': 563, 'UG/AC': 564, 'UG/AU': 565, 'UG/AN': 566, 'UG/G': 567, 'UG/GA': 568, 'UG/GG': 569, 'UG/GC': 570, 'UG/GU': 571, 'UG/GN': 572, 'UG/C': 573, 'UG/CA': 574, 'UG/CG': 575, 'UG/CC': 576, 'UG/CU': 577, 'UG/CN': 578, 'UG/U': 579, 'UG/UA': 580, 'UG/UG': 581, 'UG/UC': 582, 'UG/UU': 583, 'UG/UN': 584, 'UG/N': 0, 'UG/NA': 585, 'UG/NG': 586, 'UG/NC': 587, 'UG/NU': 588, 'UG/NN': 0, 'UC/A': 589, 'UC/AA': 590, 'UC/AG': 591, 'UC/AC': 592, 'UC/AU': 593, 'UC/AN': 594, 'UC/G': 595, 'UC/GA': 596, 'UC/GG': 597, 'UC/GC': 598, 'UC/GU': 599, 'UC/GN': 600, 'UC/C': 601, 'UC/CA': 602, 'UC/CG': 603, 'UC/CC': 604, 'UC/CU': 605, 'UC/CN': 606, 'UC/U': 607, 'UC/UA': 608, 'UC/UG': 609, 'UC/UC': 610, 'UC/UU': 611, 'UC/UN': 612, 'UC/N': 0, 'UC/NA': 613, 'UC/NG': 614, 'UC/NC': 615, 'UC/NU': 616, 'UC/NN': 0, 'UU/A': 617, 'UU/AA': 618, 'UU/AG': 619, 'UU/AC': 620, 'UU/AU': 621, 'UU/AN': 622, 'UU/G': 623, 'UU/GA': 624, 'UU/GG': 625, 'UU/GC': 626, 'UU/GU': 627, 'UU/GN': 628, 'UU/C': 629, 'UU/CA': 630, 'UU/CG': 631, 'UU/CC': 632, 'UU/CU': 633, 'UU/CN': 634, 'UU/U': 635, 'UU/UA': 636, 'UU/UG': 637, 'UU/UC': 638, 'UU/UU': 639, 'UU/UN': 640, 'UU/N': 0, 'UU/NA': 641, 'UU/NG': 642, 'UU/NC': 643, 'UU/NU': 644, 'UU/NN': 0, 'UN/A': 645, 'UN/AA': 646, 'UN/AG': 647, 'UN/AC': 648, 'UN/AU': 649, 'UN/AN': 650, 'UN/G': 651, 'UN/GA': 652, 'UN/GG': 653, 'UN/GC': 654, 'UN/GU': 655, 'UN/GN': 656, 'UN/C': 657, 'UN/CA': 658, 'UN/CG': 659, 'UN/CC': 660, 'UN/CU': 661, 'UN/CN': 662, 'UN/U': 663, 'UN/UA': 664, 'UN/UG': 665, 'UN/UC': 666, 'UN/UU': 667, 'UN/UN': 668, 'UN/N': 0, 'UN/NA': 669, 'UN/NG': 670, 'UN/NC': 671, 'UN/NU': 672, 'UN/NN': 0, 'N/A': 0, 'N/AA': 0, 'N/AG': 0, 'N/AC': 0, 'N/AU': 0, 'N/AN': 0, 'N/G': 0, 'N/GA': 0, 'N/GG': 0, 'N/GC': 0, 'N/GU': 0, 'N/GN': 0, 'N/C': 0, 'N/CA': 0, 'N/CG': 0, 'N/CC': 0, 'N/CU': 0, 'N/CN': 0, 'N/U': 0, 'N/UA': 0, 'N/UG': 0, 'N/UC': 0, 'N/UU': 0, 'N/UN': 0, 'N/N': 0, 'N/NA': 0, 'N/NG': 0, 'N/NC': 0, 'N/NU': 0, 'N/NN': 0, 'NA/A': 673, 'NA/AA': 674, 'NA/AG': 675, 'NA/AC': 676, 'NA/AU': 677, 'NA/AN': 678, 'NA/G': 679, 'NA/GA': 680, 'NA/GG': 681, 'NA/GC': 682, 'NA/GU': 683, 'NA/GN': 684, 'NA/C': 685, 'NA/CA': 686, 'NA/CG': 687, 'NA/CC': 688, 'NA/CU': 689, 'NA/CN': 690, 'NA/U': 691, 'NA/UA': 692, 'NA/UG': 693, 'NA/UC': 694, 'NA/UU': 695, 'NA/UN': 696, 'NA/N': 0, 'NA/NA': 697, 'NA/NG': 698, 'NA/NC': 699, 'NA/NU': 700, 'NA/NN': 0, 'NG/A': 701, 'NG/AA': 702, 'NG/AG': 703, 'NG/AC': 704, 'NG/AU': 705, 'NG/AN': 706, 'NG/G': 707, 'NG/GA': 708, 'NG/GG': 709, 'NG/GC': 710, 'NG/GU': 711, 'NG/GN': 712, 'NG/C': 713, 'NG/CA': 714, 'NG/CG': 715, 'NG/CC': 716, 'NG/CU': 717, 'NG/CN': 718, 'NG/U': 719, 'NG/UA': 720, 'NG/UG': 721, 'NG/UC': 722, 'NG/UU': 723, 'NG/UN': 724, 'NG/N': 0, 'NG/NA': 725, 'NG/NG': 726, 'NG/NC': 727, 'NG/NU': 728, 'NG/NN': 0, 'NC/A': 729, 'NC/AA': 730, 'NC/AG': 731, 'NC/AC': 732, 'NC/AU': 733, 'NC/AN': 734, 'NC/G': 735, 'NC/GA': 736, 'NC/GG': 737, 'NC/GC': 738, 'NC/GU': 739, 'NC/GN': 740, 'NC/C': 741, 'NC/CA': 742, 'NC/CG': 743, 'NC/CC': 744, 'NC/CU': 745, 'NC/CN': 746, 'NC/U': 747, 'NC/UA': 748, 'NC/UG': 749, 'NC/UC': 750, 'NC/UU': 751, 'NC/UN': 752, 'NC/N': 0, 'NC/NA': 753, 'NC/NG': 754, 'NC/NC': 755, 'NC/NU': 756, 'NC/NN': 0, 'NU/A': 757, 'NU/AA': 758, 'NU/AG': 759, 'NU/AC': 760, 'NU/AU': 761, 'NU/AN': 762, 'NU/G': 763, 'NU/GA': 764, 'NU/GG': 765, 'NU/GC': 766, 'NU/GU': 767, 'NU/GN': 768, 'NU/C': 769, 'NU/CA': 770, 'NU/CG': 771, 'NU/CC': 772, 'NU/CU': 773, 'NU/CN': 774, 'NU/U': 775, 'NU/UA': 776, 'NU/UG': 777, 'NU/UC': 778, 'NU/UU': 779, 'NU/UN': 780, 'NU/N': 0, 'NU/NA': 781, 'NU/NG': 782, 'NU/NC': 783, 'NU/NU': 784, 'NU/NN': 0, 'NN/A': 0, 'NN/AA': 0, 'NN/AG': 0, 'NN/AC': 0, 'NN/AU': 0, 'NN/AN': 0, 'NN/G': 0, 'NN/GA': 0, 'NN/GG': 0, 'NN/GC': 0, 'NN/GU': 0, 'NN/GN': 0, 'NN/C': 0, 'NN/CA': 0, 'NN/CG': 0, 'NN/CC': 0, 'NN/CU': 0, 'NN/CN': 0, 'NN/U': 0, 'NN/UA': 0, 'NN/UG': 0, 'NN/UC': 0, 'NN/UU': 0, 'NN/UN': 0, 'NN/N': 0, 'NN/NA': 0, 'NN/NG': 0, 'NN/NC': 0, 'NN/NU': 0, 'NN/NN': 0}