# Checks that every Literate page listed in the menu of `config.md` was executed and
# rendered by the site build. A page whose code throws during execution is not written
# to `pages/literated/`, so a missing file means the page failed.

menu_pages = [m.captures[1] for m in eachmatch(r"\"src/literate/([^\"]+)\.jl\"", read("config.md", String))]

missing_files = String[]
for name in menu_pages
    for file in ("pages/literated/$name.md", "__site/pages/literated/$name/index.html")
        isfile(file) || push!(missing_files, file)
    end
end

println("Literate pages in the menu: ", join(menu_pages, ", "))
if isempty(missing_files)
    println("All pages were executed and rendered.")
else
    println("Missing files (the corresponding pages probably failed to execute):")
    foreach(f -> println("  ", f), missing_files)
    exit(1)
end
