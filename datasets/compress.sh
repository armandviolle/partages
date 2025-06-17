#!/bin/sh

while getopts ":s:n:t:o:" opt; do
    case $opt in
        s)
            source_file="$OPTARG"
            ;;
        n)
            name="$OPTARG"
            ;;
        t) 
            target_directory="$OPTARG"
            ;;
        o) 
            override="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument"
            exit 1
    esac
done
        
# Checking that source and target directories exist
if [ ! -f "$source_file" ]; then
    echo "Source directory $source_file does not exist."
    exit 1
fi
if [ ! -d "$target_directory" ]; then
    echo "Source directory $target_directory does not exist."
    exit 1
fi

# Creating target folder's & files' path
file_name=$(basename ${source_file})
target_path="$target_directory/$name"
if [ -d $target_path ]; then
    if [ $override = true ]; then
        echo "Overriding existing folder at $target_path."
        rm -r $target_path
        mkdir $target_path
    else
        echo "Existing folder found at $target_path, no permission to override. If you want to ovveride it pass the '-o true' flag."
        exit 1
    fi
else
    echo "Creating folder at $target_path."
    mkdir $target_path
fi
target_file="$target_path/$file_name"

# Compressing files into chunks of less than 99Mo (100Mo max per file on GitHub)
gzip -c $source_file | split -b 50M - "$target_file".gz.part_
declare -i num_compressed
num_compressed=$(ls $target_path | wc -l)

# Renaming compressed files oto provide exploitable formats with gzip in python
if (($num_compressed > 1)); then
    for f in $target_path/*; do 
        filename=$(basename ${f})
        base="${filename%%.*}"
        rest="${filename#*.}"
        ext1=".${rest%%.*}"
        rest2="${rest#*.}"
        ext2=".${rest2%%.*}"
        suffix="${rest2#*.}" 
        new_name="${target}${base}_${suffix}${ext1}${ext2}"
        mv $f "$target_path/$new_name"
    done
else
    filename=$(ls $target_path)
    base="${filename%%.*}"
    rest="${filename#*.}"
    ext1=".${rest%%.*}"
    rest2="${rest#*.}"
    ext2=".${rest2%%.*}"
    mv "$target_path/$filename" "$target_path/${base}${ext1}${ext2}"
fi
